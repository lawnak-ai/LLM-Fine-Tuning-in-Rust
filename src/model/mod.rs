//! Llama-style decoder (Llama 2/3, Mistral, Qwen2/2.5, Qwen3, SmolLM, TinyLlama).

pub mod linear;
pub mod load;
pub mod ops;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

pub use linear::{Ctx, FrozenLinear, FrozenWeight, Linear, Lora};
use ops::{rms_norm, softmax_last};

use crate::util::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeStyle {
    /// Rotate (x[i], x[i + d/2]) pairs: Hugging Face layout, GGUF qwen2/qwen3.
    Half,
    /// Rotate (x[2i], x[2i+1]) pairs: GGUF llama (Q/K rows permuted by the converter).
    Interleaved,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelConfig {
    pub arch: String,
    pub name: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f64,
    pub rope_theta: f64,
    pub rope_style: RopeStyle,
    /// Linear RoPE scaling factor (1.0 = none).
    pub rope_linear_scale: f64,
    #[serde(skip)]
    pub rope_freq_factors: Option<Vec<f32>>,
    pub context_length: usize,
    pub tie_embeddings: bool,
    pub attn_bias: bool,
    pub qk_norm: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Proj {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl Proj {
    pub const ALL: [Proj; 7] = [
        Proj::Q,
        Proj::K,
        Proj::V,
        Proj::O,
        Proj::Gate,
        Proj::Up,
        Proj::Down,
    ];

    pub fn short(self) -> &'static str {
        match self {
            Proj::Q => "q",
            Proj::K => "k",
            Proj::V => "v",
            Proj::O => "o",
            Proj::Gate => "gate",
            Proj::Up => "up",
            Proj::Down => "down",
        }
    }

    pub fn parse(s: &str) -> Option<Proj> {
        let s = s.trim().to_ascii_lowercase();
        let s = s.trim_end_matches("_proj");
        Some(match s {
            "q" | "attn_q" => Proj::Q,
            "k" | "attn_k" => Proj::K,
            "v" | "attn_v" => Proj::V,
            "o" | "attn_output" => Proj::O,
            "gate" | "ffn_gate" => Proj::Gate,
            "up" | "ffn_up" => Proj::Up,
            "down" | "ffn_down" => Proj::Down,
            _ => return None,
        })
    }

    pub fn gguf(self) -> &'static str {
        match self {
            Proj::Q => "attn_q",
            Proj::K => "attn_k",
            Proj::V => "attn_v",
            Proj::O => "attn_output",
            Proj::Gate => "ffn_gate",
            Proj::Up => "ffn_up",
            Proj::Down => "ffn_down",
        }
    }

    pub fn hf(self) -> &'static str {
        match self {
            Proj::Q => "self_attn.q_proj",
            Proj::K => "self_attn.k_proj",
            Proj::V => "self_attn.v_proj",
            Proj::O => "self_attn.o_proj",
            Proj::Gate => "mlp.gate_proj",
            Proj::Up => "mlp.up_proj",
            Proj::Down => "mlp.down_proj",
        }
    }

    pub fn hf_leaf(self) -> &'static str {
        self.hf().rsplit('.').next().unwrap()
    }
}

#[derive(Clone, Debug)]
pub enum Embedding {
    Quant(Arc<QTensor>),
    Dense(Tensor),
}

impl Embedding {
    pub fn forward(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Embedding::Quant(q) => q.embedding(ids),
            Embedding::Dense(w) => {
                let mut dims = ids.dims().to_vec();
                dims.push(w.dim(1)?);
                w.index_select(&ids.flatten_all()?, 0)?
                    .to_dtype(DType::F32)?
                    .reshape(dims)
            }
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        match self {
            Embedding::Quant(q) => q.storage_size_in_bytes(),
            Embedding::Dense(t) => t.elem_count() * t.dtype().size_in_bytes(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Block {
    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl Block {
    pub fn proj(&self, p: Proj) -> &Linear {
        match p {
            Proj::Q => &self.q,
            Proj::K => &self.k,
            Proj::V => &self.v,
            Proj::O => &self.o,
            Proj::Gate => &self.gate,
            Proj::Up => &self.up,
            Proj::Down => &self.down,
        }
    }

    pub fn proj_mut(&mut self, p: Proj) -> &mut Linear {
        match p {
            Proj::Q => &mut self.q,
            Proj::K => &mut self.k,
            Proj::V => &mut self.v,
            Proj::O => &mut self.o,
            Proj::Gate => &mut self.gate,
            Proj::Up => &mut self.up,
            Proj::Down => &mut self.down,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelSource {
    Gguf { path: PathBuf },
    HfDir { path: PathBuf },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f64,
    pub dropout: f32,
    pub targets: Vec<Proj>,
}

/// Per-layer key/value cache for generation.
#[derive(Default, Clone)]
pub struct KvCache {
    pub layers: Vec<Option<(Tensor, Tensor)>>,
}

impl KvCache {
    pub fn new(n: usize) -> Self {
        Self {
            layers: vec![None; n],
        }
    }
    pub fn len(&self) -> usize {
        self.layers
            .first()
            .and_then(|l| l.as_ref())
            .map(|(k, _)| k.dim(2).unwrap_or(0))
            .unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

struct RopeTables {
    cos: Tensor,
    sin: Tensor,
    len: usize,
}

pub struct Model {
    pub cfg: ModelConfig,
    pub embed: Embedding,
    pub blocks: Vec<Block>,
    pub norm: Tensor,
    pub lm_head: FrozenLinear,
    pub device: Device,
    pub source: ModelSource,
    pub lora: Option<LoraConfig>,
    rope: Mutex<Option<RopeTables>>,
    masks: Mutex<HashMap<(usize, usize), Tensor>>,
}

impl Model {
    pub fn new(
        cfg: ModelConfig,
        embed: Embedding,
        blocks: Vec<Block>,
        norm: Tensor,
        lm_head: FrozenLinear,
        device: Device,
        source: ModelSource,
    ) -> Self {
        Self {
            cfg,
            embed,
            blocks,
            norm,
            lm_head,
            device,
            source,
            lora: None,
            rope: Mutex::new(None),
            masks: Mutex::new(HashMap::new()),
        }
    }

    /// A model with random dense weights (tests and benchmarks).
    pub fn random(cfg: ModelConfig, device: &Device, seed: u64) -> candle_core::Result<Self> {
        let mut rng = Rng::new(seed);
        let mut mat = |o: usize, i: usize, s: f32| -> candle_core::Result<Tensor> {
            let v: Vec<f32> = (0..o * i).map(|_| rng.normal() * s).collect();
            Tensor::from_vec(v, (o, i), device)
        };
        let lin = |w: Tensor, l: usize, p: Proj| {
            Linear::new(
                FrozenLinear::new(FrozenWeight::Dense(w), None),
                crate::util::mix_seed(&[l as u64, p as u64]),
            )
        };
        let (h, f, hd) = (cfg.hidden_size, cfg.intermediate_size, cfg.head_dim);
        let ones = |n: usize| Tensor::ones(n, DType::F32, device);
        let mut blocks = vec![];
        for l in 0..cfg.num_layers {
            blocks.push(Block {
                attn_norm: ones(h)?,
                ffn_norm: ones(h)?,
                q_norm: if cfg.qk_norm { Some(ones(hd)?) } else { None },
                k_norm: if cfg.qk_norm { Some(ones(hd)?) } else { None },
                q: lin(mat(cfg.num_heads * hd, h, 0.1)?, l, Proj::Q),
                k: lin(mat(cfg.num_kv_heads * hd, h, 0.1)?, l, Proj::K),
                v: lin(mat(cfg.num_kv_heads * hd, h, 0.1)?, l, Proj::V),
                o: lin(mat(h, cfg.num_heads * hd, 0.1)?, l, Proj::O),
                gate: lin(mat(f, h, 0.1)?, l, Proj::Gate),
                up: lin(mat(f, h, 0.1)?, l, Proj::Up),
                down: lin(mat(h, f, 0.1)?, l, Proj::Down),
            });
        }
        let emb = mat(cfg.vocab_size, h, 0.5)?;
        let head = FrozenLinear::new(FrozenWeight::Dense(emb.clone()), None);
        let norm = ones(h)?;
        Ok(Self::new(
            cfg,
            Embedding::Dense(emb),
            blocks,
            norm,
            head,
            device.clone(),
            ModelSource::HfDir {
                path: PathBuf::new(),
            },
        ))
    }

    pub fn is_gguf(&self) -> bool {
        matches!(self.source, ModelSource::Gguf { .. })
    }

    /// Number of base model parameters.
    pub fn num_params(&self) -> u64 {
        let c = &self.cfg;
        let attn = c.hidden_size * (c.num_heads * c.head_dim) * 2
            + c.hidden_size * (c.num_kv_heads * c.head_dim) * 2;
        let mlp = 3 * c.hidden_size * c.intermediate_size;
        let per_layer = attn + mlp + 2 * c.hidden_size;
        let emb = c.vocab_size * c.hidden_size;
        let head = if c.tie_embeddings { 0 } else { emb };
        (per_layer * c.num_layers + emb + head + c.hidden_size) as u64
    }

    pub fn weights_size_in_bytes(&self) -> usize {
        let mut n = self.embed.size_in_bytes();
        if !self.cfg.tie_embeddings {
            n += self.lm_head.weight.size_in_bytes();
        }
        for b in &self.blocks {
            for p in Proj::ALL {
                n += b.proj(p).base.weight.size_in_bytes();
            }
        }
        n
    }

    /// Histogram of weight storage types, e.g. {"q4k": 120, "q6k": 24}.
    pub fn weight_types(&self) -> HashMap<String, usize> {
        let mut h = HashMap::new();
        for b in &self.blocks {
            for p in Proj::ALL {
                *h.entry(b.proj(p).base.weight.dtype_name()).or_insert(0) += 1;
            }
        }
        h
    }

    pub fn trainable_params(&self) -> usize {
        self.lora_vars().iter().map(|v| v.elem_count()).sum()
    }

    /// Attach fresh LoRA adapters (replacing any existing ones).
    pub fn attach_lora(&mut self, cfg: LoraConfig, seed: u64) -> Result<()> {
        if cfg.rank == 0 {
            bail!("LoRA rank must be at least 1");
        }
        if cfg.targets.is_empty() {
            bail!("select at least one target module for LoRA");
        }
        let mut rng = Rng::new(seed);
        let device = self.device.clone();
        for blk in self.blocks.iter_mut() {
            for p in Proj::ALL {
                let lin = blk.proj_mut(p);
                lin.lora = if cfg.targets.contains(&p) {
                    Some(Lora::new(
                        lin.in_features(),
                        lin.out_features(),
                        cfg.rank,
                        cfg.alpha,
                        cfg.dropout,
                        &device,
                        &mut rng,
                    )?)
                } else {
                    None
                };
            }
        }
        self.lora = Some(cfg);
        Ok(())
    }

    pub fn detach_lora(&mut self) {
        for blk in self.blocks.iter_mut() {
            for p in Proj::ALL {
                blk.proj_mut(p).lora = None;
            }
        }
        self.lora = None;
    }

    pub fn set_lora_dropout(&mut self, p: f32) {
        for blk in self.blocks.iter_mut() {
            for pr in Proj::ALL {
                if let Some(l) = blk.proj_mut(pr).lora.as_mut() {
                    l.dropout = p;
                }
            }
        }
        if let Some(c) = self.lora.as_mut() {
            c.dropout = p;
        }
    }

    /// All trainable variables in a stable order: layer, projection, (A, B).
    pub fn lora_vars(&self) -> Vec<candle_core::Var> {
        let mut v = vec![];
        for blk in &self.blocks {
            for p in Proj::ALL {
                if let Some(l) = &blk.proj(p).lora {
                    v.push(l.a.clone());
                    v.push(l.b.clone());
                }
            }
        }
        v
    }

    /// Variables belonging to one layer.
    pub fn layer_lora_vars(&self, layer: usize) -> Vec<candle_core::Var> {
        let mut v = vec![];
        for p in Proj::ALL {
            if let Some(l) = &self.blocks[layer].proj(p).lora {
                v.push(l.a.clone());
                v.push(l.b.clone());
            }
        }
        v
    }

    fn rope_tables(&self, start: usize, len: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let need = start + len;
        let mut guard = self.rope.lock().unwrap();
        let rebuild = guard.as_ref().map(|t| t.len < need).unwrap_or(true);
        if rebuild {
            let cap = need.max(512).next_power_of_two();
            let half = self.cfg.head_dim / 2;
            let mut inv = Vec::with_capacity(half);
            for i in 0..half {
                let mut f = 1.0
                    / self
                        .cfg
                        .rope_theta
                        .powf((2 * i) as f64 / self.cfg.head_dim as f64);
                if let Some(ff) = &self.cfg.rope_freq_factors {
                    if let Some(x) = ff.get(i) {
                        f /= *x as f64;
                    }
                }
                inv.push(f / self.cfg.rope_linear_scale);
            }
            let mut cos = Vec::with_capacity(cap * half);
            let mut sin = Vec::with_capacity(cap * half);
            for p in 0..cap {
                for f in &inv {
                    let a = p as f64 * f;
                    cos.push(a.cos() as f32);
                    sin.push(a.sin() as f32);
                }
            }
            let cos = Tensor::from_vec(cos, (cap, half), &self.device)?;
            let sin = Tensor::from_vec(sin, (cap, half), &self.device)?;
            *guard = Some(RopeTables { cos, sin, len: cap });
        }
        let t = guard.as_ref().unwrap();
        Ok((t.cos.narrow(0, start, len)?, t.sin.narrow(0, start, len)?))
    }

    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        ops::rope(x, cos, sin, self.cfg.rope_style == RopeStyle::Interleaved)
    }

    fn causal_mask(&self, t: usize, past: usize) -> candle_core::Result<Tensor> {
        let mut masks = self.masks.lock().unwrap();
        if let Some(m) = masks.get(&(t, past)) {
            return Ok(m.clone());
        }
        let s = t + past;
        let mut v = vec![0f32; t * s];
        for i in 0..t {
            for j in (past + i + 1)..s {
                v[i * s + j] = f32::NEG_INFINITY;
            }
        }
        let m = Tensor::from_vec(v, (t, s), &self.device)?;
        if masks.len() > 64 {
            masks.clear();
        }
        masks.insert((t, past), m.clone());
        Ok(m)
    }

    pub fn embed(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        self.embed.forward(ids)
    }

    /// One transformer block. `x` is (batch, seq, hidden).
    pub fn block_forward(
        &self,
        li: usize,
        x: &Tensor,
        ctx: &Ctx,
        cache: Option<&mut Option<(Tensor, Tensor)>>,
    ) -> candle_core::Result<Tensor> {
        let c = &self.cfg;
        let blk = &self.blocks[li];
        let (b, t, _) = x.dims3()?;
        let past = cache
            .as_ref()
            .and_then(|c| c.as_ref())
            .map(|(k, _)| k.dim(2).unwrap_or(0))
            .unwrap_or(0);

        let h = rms_norm(x, &blk.attn_norm, c.rms_eps)?;
        let q = blk
            .q
            .forward(&h, ctx)?
            .reshape((b, t, c.num_heads, c.head_dim))?;
        let k = blk
            .k
            .forward(&h, ctx)?
            .reshape((b, t, c.num_kv_heads, c.head_dim))?;
        let v = blk
            .v
            .forward(&h, ctx)?
            .reshape((b, t, c.num_kv_heads, c.head_dim))?;
        let q = match &blk.q_norm {
            Some(w) => rms_norm(&q, w, c.rms_eps)?,
            None => q,
        };
        let k = match &blk.k_norm {
            Some(w) => rms_norm(&k, w, c.rms_eps)?,
            None => k,
        };
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let (cos, sin) = self.rope_tables(past, t)?;
        let q = self.apply_rope(&q, &cos, &sin)?;
        let k = self.apply_rope(&k, &cos, &sin)?;

        let (k, v) = match cache {
            Some(slot) => {
                let (k, v) = match slot.take() {
                    Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?),
                    None => (k, v),
                };
                *slot = Some((k.clone(), v.clone()));
                (k, v)
            }
            None => (k, v),
        };
        let rep = c.num_heads / c.num_kv_heads;
        let (k, v) = if rep > 1 {
            let s = k.dim(2)?;
            let expand = |x: &Tensor| -> candle_core::Result<Tensor> {
                x.unsqueeze(2)?
                    .broadcast_as((b, c.num_kv_heads, rep, s, c.head_dim))?
                    .reshape((b, c.num_heads, s, c.head_dim))
            };
            (expand(&k)?, expand(&v)?)
        } else {
            (k, v)
        };
        let scale = 1.0 / (c.head_dim as f64).sqrt();
        let att = q.matmul(&k.t()?)?.affine(scale, 0.0)?;
        let att = if t > 1 {
            att.broadcast_add(&self.causal_mask(t, past)?)?
        } else {
            att
        };
        let att = softmax_last(&att)?;
        let out = att
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, t, c.num_heads * c.head_dim))?;
        let x = x.add(&blk.o.forward(&out, ctx)?)?;

        let h = rms_norm(&x, &blk.ffn_norm, c.rms_eps)?;
        let g = blk.gate.forward(&h, ctx)?.silu()?;
        let u = blk.up.forward(&h, ctx)?;
        let m = blk.down.forward(&g.mul(&u)?, ctx)?;
        x.add(&m)
    }

    pub fn final_norm(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        rms_norm(x, &self.norm, self.cfg.rms_eps)
    }

    pub fn logits(&self, h: &Tensor) -> candle_core::Result<Tensor> {
        self.lm_head.forward(h)
    }

    /// Full forward pass (no cache). Returns logits (batch, seq, vocab).
    pub fn forward(&self, ids: &Tensor, ctx: &Ctx) -> candle_core::Result<Tensor> {
        let mut x = self.embed(ids)?;
        for li in 0..self.blocks.len() {
            x = self.block_forward(li, &x, ctx, None)?;
        }
        self.logits(&self.final_norm(&x)?)
    }

    /// Incremental forward for generation. Returns logits of the last position (vocab,).
    pub fn forward_cached(&self, ids: &[u32], cache: &mut KvCache) -> candle_core::Result<Tensor> {
        if cache.layers.len() != self.blocks.len() {
            *cache = KvCache::new(self.blocks.len());
        }
        let ctx = Ctx::inference();
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let mut x = self.embed(&input)?;
        for li in 0..self.blocks.len() {
            x = self.block_forward(li, &x, &ctx, Some(&mut cache.layers[li]))?;
        }
        let t = x.dim(1)?;
        let last = x.narrow(1, t - 1, 1)?;
        self.logits(&self.final_norm(&last)?)?.flatten_all()
    }
}
