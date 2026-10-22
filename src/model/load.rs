//! Loading models from GGUF files or Hugging Face safetensors directories.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use serde_json::Value as Json;

use super::{Block, Embedding, FrozenLinear, FrozenWeight, Linear, Model, ModelConfig, ModelSource, Proj, RopeStyle};
use crate::tokenizer::Tokenizer;
use crate::util::mix_seed;

/// How the frozen base weights are stored in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasePrecision {
    /// Keep what the file has (quantized GGUF stays quantized: QLoRA).
    Auto,
    F32,
    F16,
    BF16,
    /// Quantize on load (useful for safetensors models).
    Quant(GgmlDType),
}

impl BasePrecision {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" | "keep" => Self::Auto,
            "f32" | "fp32" | "float32" => Self::F32,
            "f16" | "fp16" | "float16" => Self::F16,
            "bf16" | "bfloat16" => Self::BF16,
            other => Self::Quant(parse_ggml_dtype(other)?),
        })
    }
}

pub fn parse_ggml_dtype(s: &str) -> Result<GgmlDType> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "f32" => GgmlDType::F32,
        "f16" => GgmlDType::F16,
        "bf16" => GgmlDType::BF16,
        "q4_0" => GgmlDType::Q4_0,
        "q4_1" => GgmlDType::Q4_1,
        "q5_0" => GgmlDType::Q5_0,
        "q5_1" => GgmlDType::Q5_1,
        "q8_0" => GgmlDType::Q8_0,
        "q2_k" | "q2k" => GgmlDType::Q2K,
        "q3_k" | "q3k" => GgmlDType::Q3K,
        "q4_k" | "q4k" | "q4_k_m" | "q4_k_s" => GgmlDType::Q4K,
        "q5_k" | "q5k" | "q5_k_m" | "q5_k_s" => GgmlDType::Q5K,
        "q6_k" | "q6k" => GgmlDType::Q6K,
        other => bail!("unknown quantization type '{other}'"),
    })
}

/// Load a model and its tokenizer from a `.gguf` file or a Hugging Face directory.
pub fn load_model(
    path: &Path,
    device: &Device,
    precision: BasePrecision,
    progress: &dyn Fn(String),
) -> Result<(Model, Tokenizer)> {
    if !path.exists() {
        bail!("path does not exist: {}", path.display());
    }
    if path.is_dir() {
        if path.join("config.json").exists() {
            return load_hf(path, device, precision, progress);
        }
        let ggufs: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false))
            .collect();
        if ggufs.len() == 1 {
            return load_gguf(&ggufs[0], device, precision, progress);
        }
        bail!("{} has no config.json and not exactly one .gguf file", path.display());
    }
    let ext = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "gguf" => load_gguf(path, device, precision, progress),
        "safetensors" | "json" => load_hf(path.parent().context("no parent dir")?, device, precision, progress),
        _ => bail!("unsupported model file {} (expected .gguf or a Hugging Face folder)", path.display()),
    }
}

fn md_usize(md: &HashMap<String, gguf_file::Value>, key: &str) -> Option<usize> {
    use gguf_file::Value as V;
    match md.get(key)? {
        V::U8(x) => Some(*x as usize),
        V::I8(x) => Some(*x as usize),
        V::U16(x) => Some(*x as usize),
        V::I16(x) => Some(*x as usize),
        V::U32(x) => Some(*x as usize),
        V::I32(x) => Some(*x as usize),
        V::U64(x) => Some(*x as usize),
        V::I64(x) => Some(*x as usize),
        V::Array(a) => a.first().and_then(|v| match v {
            V::U32(x) => Some(*x as usize),
            V::I32(x) => Some(*x as usize),
            _ => None,
        }),
        _ => None,
    }
}

fn md_f64(md: &HashMap<String, gguf_file::Value>, key: &str) -> Option<f64> {
    use gguf_file::Value as V;
    match md.get(key)? {
        V::F32(x) => Some(*x as f64),
        V::F64(x) => Some(*x),
        _ => md_usize(md, key).map(|x| x as f64),
    }
}

fn md_str(md: &HashMap<String, gguf_file::Value>, key: &str) -> Option<String> {
    md.get(key).and_then(|v| v.to_string().ok().cloned())
}

/// Convert a loaded tensor into the requested storage for a frozen matrix.
fn frozen_from_qtensor(qt: QTensor, precision: BasePrecision, device: &Device) -> Result<FrozenWeight> {
    let src = qt.dtype();
    let dense_src = matches!(src, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16);
    Ok(match precision {
        BasePrecision::Auto if !dense_src => FrozenWeight::Quant(Arc::new(qt)),
        BasePrecision::Auto => FrozenWeight::Dense(match src {
            GgmlDType::F16 => qt.dequantize_f16(device)?,
            GgmlDType::BF16 => qt.dequantize(device)?.to_dtype(DType::BF16)?,
            _ => qt.dequantize(device)?,
        }),
        BasePrecision::F32 => FrozenWeight::Dense(qt.dequantize(device)?),
        BasePrecision::F16 => FrozenWeight::Dense(qt.dequantize_f16(device)?),
        BasePrecision::BF16 => FrozenWeight::Dense(qt.dequantize(device)?.to_dtype(DType::BF16)?),
        BasePrecision::Quant(dt) if dt == src => FrozenWeight::Quant(Arc::new(qt)),
        BasePrecision::Quant(dt) => {
            let t = qt.dequantize(&Device::Cpu)?;
            frozen_from_dense(t, BasePrecision::Quant(dt), device)?
        }
    })
}

fn frozen_from_dense(t: Tensor, precision: BasePrecision, device: &Device) -> Result<FrozenWeight> {
    Ok(match precision {
        BasePrecision::Auto => FrozenWeight::Dense(t.to_device(device)?),
        BasePrecision::F32 => FrozenWeight::Dense(t.to_dtype(DType::F32)?.to_device(device)?),
        BasePrecision::F16 => FrozenWeight::Dense(t.to_dtype(DType::F16)?.to_device(device)?),
        BasePrecision::BF16 => FrozenWeight::Dense(t.to_dtype(DType::BF16)?.to_device(device)?),
        BasePrecision::Quant(dt) => {
            let cols = t.dim(1)?;
            let dt = if cols % dt.block_size() == 0 {
                dt
            } else if cols % GgmlDType::Q8_0.block_size() == 0 {
                GgmlDType::Q8_0
            } else {
                return Ok(FrozenWeight::Dense(t.to_dtype(DType::F16)?.to_device(device)?));
            };
            let cpu = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
            FrozenWeight::Quant(Arc::new(QTensor::quantize_onto(&cpu, dt, device)?))
        }
    })
}

fn check_dims(name: &str, w: &FrozenWeight, out: usize, inp: usize) -> Result<()> {
    let (o, i) = w.dims();
    if (o, i) != (out, inp) {
        bail!("tensor {name} has shape ({o}, {i}), expected ({out}, {inp})");
    }
    Ok(())
}

pub fn load_gguf(path: &Path, device: &Device, precision: BasePrecision, progress: &dyn Fn(String)) -> Result<(Model, Tokenizer)> {
    let mut file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let content = gguf_file::Content::read(&mut file).with_context(|| format!("{} is not a valid GGUF file", path.display()))?;
    let md = &content.metadata;
    let arch = md_str(md, "general.architecture").unwrap_or_else(|| "llama".to_string());
    let (rope_style, qk_norm) = match arch.as_str() {
        "llama" | "mistral" => (RopeStyle::Interleaved, false),
        "qwen2" => (RopeStyle::Half, false),
        "qwen3" => (RopeStyle::Half, true),
        other => bail!(
            "model architecture '{other}' is not supported yet (supported: llama, mistral, qwen2, qwen3 \
             — this covers Llama 2/3, Mistral, TinyLlama, SmolLM, Qwen 2/2.5/3 and their fine-tunes)"
        ),
    };
    if md.get("general.type").and_then(|v| v.to_string().ok()).map(|s| s == "adapter").unwrap_or(false) {
        bail!("{} is a LoRA adapter, not a base model", path.display());
    }
    let key = |k: &str| format!("{arch}.{k}");
    let hidden = md_usize(md, &key("embedding_length")).context("missing embedding_length")?;
    let layers = md_usize(md, &key("block_count")).context("missing block_count")?;
    let ff = md_usize(md, &key("feed_forward_length")).context("missing feed_forward_length")?;
    let heads = md_usize(md, &key("attention.head_count")).context("missing head_count")?;
    let kv_heads = md_usize(md, &key("attention.head_count_kv")).unwrap_or(heads);
    let head_dim = md_usize(md, &key("attention.key_length")).unwrap_or(hidden / heads);
    let eps = md_f64(md, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);
    let theta = md_f64(md, &key("rope.freq_base")).unwrap_or(10_000.0);
    let ctx_len = md_usize(md, &key("context_length")).unwrap_or(4096);
    if let Some(rd) = md_usize(md, &key("rope.dimension_count")) {
        if rd != head_dim {
            bail!("partial rotary embeddings (rope dim {rd} != head dim {head_dim}) are not supported");
        }
    }
    let mut rope_linear_scale = 1.0;
    match md_str(md, &key("rope.scaling.type")).as_deref() {
        Some("linear") => {
            rope_linear_scale = md_f64(md, &key("rope.scaling.factor")).unwrap_or(1.0);
        }
        Some("yarn") => progress("warning: YaRN rope scaling is ignored (fine for sequences inside the original context)".into()),
        _ => {}
    }

    let tokenizer = Tokenizer::from_gguf(md)?;
    progress(format!("{arch}: {layers} layers, hidden {hidden}, {heads} heads ({kv_heads} kv)"));

    let mut read = |name: &str| -> Result<QTensor> {
        content.tensor(&mut file, name, device).with_context(|| format!("cannot read tensor {name}"))
    };
    let has = |name: &str| content.tensor_infos.contains_key(name);
    let dense_f32 = |q: QTensor| -> Result<Tensor> { Ok(q.dequantize(device)?.to_dtype(DType::F32)?) };

    let embd_q = read("token_embd.weight")?;
    let vocab = embd_q.shape().dims()[0];
    let embd_dtype = embd_q.dtype();
    let embd_dense = matches!(embd_dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16);
    let block_ok = hidden % embd_dtype.block_size() == 0;
    let embd_arc = Arc::new(embd_q);
    let embed = match precision {
        BasePrecision::Auto | BasePrecision::Quant(_) if !embd_dense && block_ok => Embedding::Quant(embd_arc.clone()),
        BasePrecision::Auto => Embedding::Dense(match embd_dtype {
            GgmlDType::F16 => embd_arc.dequantize_f16(device)?,
            GgmlDType::BF16 => embd_arc.dequantize(device)?.to_dtype(DType::BF16)?,
            _ => embd_arc.dequantize(device)?,
        }),
        BasePrecision::F32 => Embedding::Dense(embd_arc.dequantize(device)?),
        BasePrecision::BF16 => Embedding::Dense(embd_arc.dequantize(device)?.to_dtype(DType::BF16)?),
        BasePrecision::F16 => Embedding::Dense(embd_arc.dequantize_f16(device)?),
        BasePrecision::Quant(dt) if hidden % dt.block_size() == 0 => {
            let cpu = embd_arc.dequantize(&Device::Cpu)?;
            Embedding::Quant(Arc::new(QTensor::quantize_onto(&cpu, dt, device)?))
        }
        BasePrecision::Quant(_) => Embedding::Dense(embd_arc.dequantize_f16(device)?),
    };
    let tie = !has("output.weight");
    let lm_head_w = if tie {
        match (&embed, precision) {
            (Embedding::Quant(q), _) => FrozenWeight::Quant(q.clone()),
            (Embedding::Dense(t), BasePrecision::Quant(_)) => frozen_from_dense(t.clone(), precision, device)?,
            (Embedding::Dense(t), _) => FrozenWeight::Dense(t.clone()),
        }
    } else {
        frozen_from_qtensor(read("output.weight")?, precision, device)?
    };
    check_dims("output.weight", &lm_head_w, vocab, hidden)?;
    let lm_head = FrozenLinear::new(lm_head_w, None);
    let norm = dense_f32(read("output_norm.weight")?)?;
    let rope_freq_factors = if has("rope_freqs.weight") {
        Some(read("rope_freqs.weight")?.dequantize(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?)
    } else {
        None
    };

    let mut blocks = Vec::with_capacity(layers);
    let mut attn_bias = false;
    for i in 0..layers {
        progress(format!("loading layer {}/{}", i + 1, layers));
        let p = |s: &str| format!("blk.{i}.{s}");
        let lin = |proj: Proj, out: usize, inp: usize, read: &mut dyn FnMut(&str) -> Result<QTensor>| -> Result<Linear> {
            let wname = p(&format!("{}.weight", proj.gguf()));
            let w = frozen_from_qtensor(read(&wname)?, precision, device)?;
            check_dims(&wname, &w, out, inp)?;
            let bname = p(&format!("{}.bias", proj.gguf()));
            let bias = if has(&bname) { Some(read(&bname)?.dequantize(device)?) } else { None };
            Ok(Linear::new(FrozenLinear::new(w, bias), mix_seed(&[i as u64, proj as u64])))
        };
        let q = lin(Proj::Q, heads * head_dim, hidden, &mut read)?;
        let k = lin(Proj::K, kv_heads * head_dim, hidden, &mut read)?;
        let v = lin(Proj::V, kv_heads * head_dim, hidden, &mut read)?;
        let o = lin(Proj::O, hidden, heads * head_dim, &mut read)?;
        let gate = lin(Proj::Gate, ff, hidden, &mut read)?;
        let up = lin(Proj::Up, ff, hidden, &mut read)?;
        let down = lin(Proj::Down, hidden, ff, &mut read)?;
        attn_bias |= q.base.bias.is_some();
        let q_norm = if has(&p("attn_q_norm.weight")) { Some(dense_f32(read(&p("attn_q_norm.weight"))?)?) } else { None };
        let k_norm = if has(&p("attn_k_norm.weight")) { Some(dense_f32(read(&p("attn_k_norm.weight"))?)?) } else { None };
        if qk_norm && (q_norm.is_none() || k_norm.is_none()) {
            bail!("qwen3 model without attn_q_norm/attn_k_norm tensors");
        }
        blocks.push(Block {
            attn_norm: dense_f32(read(&p("attn_norm.weight"))?)?,
            ffn_norm: dense_f32(read(&p("ffn_norm.weight"))?)?,
            q_norm,
            k_norm,
            q,
            k,
            v,
            o,
            gate,
            up,
            down,
        });
    }
    let name = md_str(md, "general.name").unwrap_or_else(|| {
        path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()
    });
    let cfg = ModelConfig {
        arch,
        name,
        vocab_size: vocab,
        hidden_size: hidden,
        intermediate_size: ff,
        num_layers: layers,
        num_heads: heads,
        num_kv_heads: kv_heads,
        head_dim,
        rms_eps: eps,
        rope_theta: theta,
        rope_style,
        rope_linear_scale,
        rope_freq_factors,
        context_length: ctx_len,
        tie_embeddings: tie,
        attn_bias,
        qk_norm,
    };
    let model = Model::new(cfg, embed, blocks, norm, lm_head, device.clone(), ModelSource::Gguf { path: path.to_path_buf() });
    Ok((model, tokenizer))
}

/// Safetensors files of a Hugging Face model directory.
pub fn hf_weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let j: Json = serde_json::from_str(&std::fs::read_to_string(&index)?)?;
        let mut files: Vec<String> = j["weight_map"]
            .as_object()
            .context("weight_map missing in index")?
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        files.sort();
        files.dedup();
        return Ok(files.into_iter().map(|f| dir.join(f)).collect());
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
        .filter(|p| !p.file_name().map(|n| n.to_string_lossy().starts_with("adapter")).unwrap_or(false))
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no .safetensors files in {}", dir.display());
    }
    Ok(files)
}

/// Llama 3 style RoPE frequency factors (inv_freq / scaled_inv_freq).
fn llama3_freq_factors(head_dim: usize, theta: f64, scaling: &Json) -> Vec<f32> {
    let factor = scaling["factor"].as_f64().unwrap_or(8.0);
    let low = scaling["low_freq_factor"].as_f64().unwrap_or(1.0);
    let high = scaling["high_freq_factor"].as_f64().unwrap_or(4.0);
    let old_ctx = scaling["original_max_position_embeddings"].as_f64().unwrap_or(8192.0);
    let low_wl = old_ctx / low;
    let high_wl = old_ctx / high;
    (0..head_dim / 2)
        .map(|i| {
            let inv = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
            let wl = 2.0 * std::f64::consts::PI / inv;
            let scaled = if wl < high_wl {
                inv
            } else if wl > low_wl {
                inv / factor
            } else {
                let smooth = (old_ctx / wl - low) / (high - low);
                (1.0 - smooth) * inv / factor + smooth * inv
            };
            (inv / scaled) as f32
        })
        .collect()
}

pub fn load_hf(dir: &Path, device: &Device, precision: BasePrecision, progress: &dyn Fn(String)) -> Result<(Model, Tokenizer)> {
    let cfg: Json = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).context("cannot read config.json")?)?;
    let model_type = cfg["model_type"].as_str().unwrap_or("llama").to_string();
    let (arch, qk_norm, default_bias) = match model_type.as_str() {
        "llama" | "mistral" => ("llama", false, false),
        "qwen2" => ("qwen2", false, true),
        "qwen3" => ("qwen3", true, false),
        other => bail!("model_type '{other}' is not supported yet (supported: llama, mistral, qwen2, qwen3)"),
    };
    let get = |k: &str| cfg[k].as_u64().map(|x| x as usize);
    let hidden = get("hidden_size").context("hidden_size missing")?;
    let layers = get("num_hidden_layers").context("num_hidden_layers missing")?;
    let ff = get("intermediate_size").context("intermediate_size missing")?;
    let heads = get("num_attention_heads").context("num_attention_heads missing")?;
    let kv_heads = get("num_key_value_heads").unwrap_or(heads);
    let head_dim = get("head_dim").unwrap_or(hidden / heads);
    let eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6);
    let rope_params = if cfg["rope_parameters"].is_object() { &cfg["rope_parameters"] } else { &cfg["rope_scaling"] };
    let theta = cfg["rope_theta"].as_f64().or_else(|| cfg["rope_parameters"]["rope_theta"].as_f64()).unwrap_or(10_000.0);
    let ctx_len = get("max_position_embeddings").unwrap_or(4096);
    let tie = cfg["tie_word_embeddings"].as_bool().unwrap_or(false);
    let attn_bias = cfg["attention_bias"].as_bool().unwrap_or(default_bias);
    if cfg["mlp_bias"].as_bool().unwrap_or(false) {
        bail!("models with MLP biases are not supported");
    }
    let mut rope_linear_scale = 1.0;
    let mut rope_freq_factors = None;
    if rope_params.is_object() {
        let rt = rope_params["rope_type"].as_str().or_else(|| rope_params["type"].as_str()).unwrap_or("default");
        match rt {
            "llama3" => rope_freq_factors = Some(llama3_freq_factors(head_dim, theta, rope_params)),
            "linear" => rope_linear_scale = rope_params["factor"].as_f64().unwrap_or(1.0),
            "default" => {}
            other => progress(format!("warning: rope scaling '{other}' is ignored")),
        }
    }
    let tokenizer = Tokenizer::from_hf_dir(dir)?;
    progress(format!("{model_type}: {layers} layers, hidden {hidden}, {heads} heads ({kv_heads} kv)"));

    let files = hf_weight_files(dir)?;
    // SAFETY: the files are memory mapped read-only and not modified while loaded.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&files)? };
    let names: std::collections::HashSet<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
    let load = |name: &str| -> Result<Tensor> { Ok(st.load(name, &Device::Cpu).with_context(|| format!("missing tensor {name}"))?) };
    let small = |name: &str| -> Result<Tensor> { Ok(load(name)?.to_dtype(DType::F32)?.to_device(device)?) };

    let embd = load("model.embed_tokens.weight")?;
    let vocab = embd.dim(0)?;
    let has_head = names.contains("lm_head.weight") && !tie;
    let embed = match precision {
        BasePrecision::Auto => Embedding::Dense(embd.to_device(device)?),
        BasePrecision::F32 => Embedding::Dense(embd.to_dtype(DType::F32)?.to_device(device)?),
        BasePrecision::BF16 => Embedding::Dense(embd.to_dtype(DType::BF16)?.to_device(device)?),
        BasePrecision::F16 => Embedding::Dense(embd.to_dtype(DType::F16)?.to_device(device)?),
        BasePrecision::Quant(dt) if hidden % dt.block_size() == 0 => {
            let cpu = embd.to_dtype(DType::F32)?;
            Embedding::Quant(Arc::new(QTensor::quantize_onto(&cpu, dt, device)?))
        }
        BasePrecision::Quant(_) => Embedding::Dense(embd.to_dtype(DType::F16)?.to_device(device)?),
    };
    let lm_head_w = if has_head {
        frozen_from_dense(load("lm_head.weight")?, precision, device)?
    } else {
        match &embed {
            Embedding::Quant(q) => FrozenWeight::Quant(q.clone()),
            Embedding::Dense(t) => FrozenWeight::Dense(t.clone()),
        }
    };
    let lm_head = FrozenLinear::new(lm_head_w, None);
    let norm = small("model.norm.weight")?;

    let mut blocks = Vec::with_capacity(layers);
    for i in 0..layers {
        progress(format!("loading layer {}/{}", i + 1, layers));
        let p = |s: &str| format!("model.layers.{i}.{s}");
        let lin = |proj: Proj, out: usize, inp: usize| -> Result<Linear> {
            let wname = p(&format!("{}.weight", proj.hf()));
            let w = frozen_from_dense(load(&wname)?, precision, device)?;
            check_dims(&wname, &w, out, inp)?;
            let bname = p(&format!("{}.bias", proj.hf()));
            let bias = if names.contains(&bname) { Some(small(&bname)?) } else { None };
            Ok(Linear::new(FrozenLinear::new(w, bias), mix_seed(&[i as u64, proj as u64])))
        };
        let q_norm = if qk_norm { Some(small(&p("self_attn.q_norm.weight"))?) } else { None };
        let k_norm = if qk_norm { Some(small(&p("self_attn.k_norm.weight"))?) } else { None };
        blocks.push(Block {
            attn_norm: small(&p("input_layernorm.weight"))?,
            ffn_norm: small(&p("post_attention_layernorm.weight"))?,
            q_norm,
            k_norm,
            q: lin(Proj::Q, heads * head_dim, hidden)?,
            k: lin(Proj::K, kv_heads * head_dim, hidden)?,
            v: lin(Proj::V, kv_heads * head_dim, hidden)?,
            o: lin(Proj::O, hidden, heads * head_dim)?,
            gate: lin(Proj::Gate, ff, hidden)?,
            up: lin(Proj::Up, ff, hidden)?,
            down: lin(Proj::Down, hidden, ff)?,
        });
    }
    let name = cfg["_name_or_path"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(String::from)
        .unwrap_or_else(|| dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default());
    let mcfg = ModelConfig {
        arch: arch.to_string(),
        name,
        vocab_size: vocab,
        hidden_size: hidden,
        intermediate_size: ff,
        num_layers: layers,
        num_heads: heads,
        num_kv_heads: kv_heads,
        head_dim,
        rms_eps: eps,
        rope_theta: theta,
        rope_style: RopeStyle::Half,
        rope_linear_scale,
        rope_freq_factors,
        context_length: ctx_len,
        tie_embeddings: !has_head,
        attn_bias,
        qk_norm,
    };
    Ok((
        Model::new(mcfg, embed, blocks, norm, lm_head, device.clone(), ModelSource::HfDir { path: dir.to_path_buf() }),
        tokenizer,
    ))
}
