//! Saving and loading adapters (PEFT safetensors and llama.cpp GGUF) and exporting
//! merged models as GGUF or safetensors.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};

use crate::gguf_writer::{file_type_for, tensor_nbytes, write_gguf, TensorPlan};
use crate::model::load::hf_weight_files;
use crate::model::{Lora, LoraConfig, Model, ModelSource, Proj, RopeStyle};
use crate::tokenizer::{Tokenizer, TokenizerKind};

/// Reorder rows from llama.cpp's interleaved Q/K layout to the Hugging Face layout.
fn rows_gguf_to_hf(t: &Tensor, heads: usize) -> Result<Tensor> {
    let (out, r) = t.dims2()?;
    let d = out / heads;
    Ok(t.reshape((heads, d / 2, 2, r))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((out, r))?)
}

/// Reorder rows from the Hugging Face Q/K layout to llama.cpp's interleaved layout.
fn rows_hf_to_gguf(t: &Tensor, heads: usize) -> Result<Tensor> {
    let (out, r) = t.dims2()?;
    let d = out / heads;
    Ok(t.reshape((heads, 2, d / 2, r))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((out, r))?)
}

fn qk_heads(model: &Model, p: Proj) -> Option<usize> {
    match p {
        Proj::Q => Some(model.cfg.num_heads),
        Proj::K => Some(model.cfg.num_kv_heads),
        _ => None,
    }
}

/// GGUF architecture name used by llama.cpp for this model.
fn gguf_arch(model: &Model) -> String {
    match model.cfg.arch.as_str() {
        "mistral" => "llama".into(),
        a => a.into(),
    }
}

/// Whether llama.cpp stores this model's Q/K rows permuted.
fn gguf_permutes_qk(model: &Model) -> bool {
    gguf_arch(model) == "llama"
}

pub fn tensor_bytes(t: &Tensor) -> Result<Vec<u8>> {
    let t = t.flatten_all()?.to_device(&Device::Cpu)?;
    Ok(match t.dtype() {
        DType::F32 => t
            .to_vec1::<f32>()?
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        DType::F16 => t
            .to_vec1::<half::f16>()?
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        DType::BF16 => t
            .to_vec1::<half::bf16>()?
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        other => bail!("cannot serialise dtype {other:?}"),
    })
}

/// Quantize / convert a dense f32 (cpu) matrix into raw GGUF bytes.
fn encode_ggml(t: &Tensor, dtype: GgmlDType) -> Result<Vec<u8>> {
    match dtype {
        GgmlDType::F32 => tensor_bytes(&t.to_dtype(DType::F32)?),
        GgmlDType::F16 => tensor_bytes(&t.to_dtype(DType::F16)?),
        GgmlDType::BF16 => tensor_bytes(&t.to_dtype(DType::BF16)?),
        q => Ok(QTensor::quantize(&t.to_dtype(DType::F32)?, q)?
            .data()?
            .to_vec()),
    }
}

/// Pick a quantization that works for a row length, like llama.cpp does.
fn fit_dtype(want: GgmlDType, row: usize) -> GgmlDType {
    if row % want.block_size() == 0 {
        return want;
    }
    let fallback = match want {
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K => GgmlDType::Q5_0,
        GgmlDType::Q5K => GgmlDType::Q5_1,
        _ => GgmlDType::Q8_0,
    };
    if row % fallback.block_size() == 0 {
        fallback
    } else {
        GgmlDType::F16
    }
}

pub fn parse_export_quant(s: &str) -> Result<Option<GgmlDType>> {
    match s.trim().to_ascii_lowercase().as_str() {
        "" | "same" | "keep" | "original" => Ok(None),
        other => Ok(Some(crate::model::load::parse_ggml_dtype(other)?)),
    }
}

/// (layer, projection, adapter)
type AdapterList<'a> = Vec<(usize, Proj, &'a Lora)>;

fn adapters(model: &Model) -> Result<(&LoraConfig, AdapterList<'_>)> {
    let cfg = model
        .lora
        .as_ref()
        .context("no LoRA adapter loaded, train or load one first")?;
    let mut v = vec![];
    for (i, b) in model.blocks.iter().enumerate() {
        for p in Proj::ALL {
            if let Some(l) = &b.proj(p).lora {
                v.push((i, p, l));
            }
        }
    }
    Ok((cfg, v))
}

/// Save a Hugging Face PEFT adapter. Returns the written files.
pub fn save_peft_adapter(model: &Model, tok: &Tokenizer, dir: &Path) -> Result<Vec<PathBuf>> {
    let (cfg, list) = adapters(model)?;
    std::fs::create_dir_all(dir)?;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (i, p, l) in list {
        let a = l.a.as_tensor().to_device(&Device::Cpu)?;
        let mut b = l.b.as_tensor().to_device(&Device::Cpu)?;
        if model.cfg.rope_style == RopeStyle::Interleaved {
            if let Some(h) = qk_heads(model, p) {
                b = rows_gguf_to_hf(&b, h)?;
            }
        }
        let base = format!("base_model.model.model.layers.{i}.{}", p.hf());
        tensors.insert(format!("{base}.lora_A.weight"), a);
        tensors.insert(format!("{base}.lora_B.weight"), b);
    }
    let st_path = dir.join("adapter_model.safetensors");
    candle_core::safetensors::save(&tensors, &st_path)?;
    let targets: Vec<&str> = cfg.targets.iter().map(|p| p.hf_leaf()).collect();
    let base_name = match &model.source {
        ModelSource::HfDir { path } => path.display().to_string(),
        ModelSource::Gguf { .. } => model.cfg.name.clone(),
    };
    let conf = serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_name,
        "r": cfg.rank,
        "lora_alpha": cfg.alpha,
        "lora_dropout": cfg.dropout,
        "target_modules": targets,
        "bias": "none",
        "fan_in_fan_out": false,
        "inference_mode": true,
        "init_lora_weights": true,
        "use_rslora": false,
        "use_dora": false,
        "modules_to_save": null,
        "layers_to_transform": null,
        "rank_pattern": {},
        "alpha_pattern": {},
    });
    let cfg_path = dir.join("adapter_config.json");
    std::fs::write(&cfg_path, serde_json::to_string_pretty(&conf)?)?;
    let mut files = vec![st_path, cfg_path];
    if let Some(t) = &tok.chat_template {
        let p = dir.join("chat_template.jinja");
        std::fs::write(&p, t)?;
        files.push(p);
    }
    Ok(files)
}

/// Save a llama.cpp LoRA adapter (GGUF).
pub fn save_gguf_adapter(model: &Model, path: &Path) -> Result<PathBuf> {
    let (cfg, list) = adapters(model)?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let permute = gguf_permutes_qk(model) && model.cfg.rope_style == RopeStyle::Half;
    let mut plans = vec![];
    let mut data = vec![];
    for (i, p, l) in list {
        let a = l.a.as_tensor().to_device(&Device::Cpu)?;
        let mut b = l.b.as_tensor().to_device(&Device::Cpu)?;
        if permute {
            if let Some(h) = qk_heads(model, p) {
                b = rows_hf_to_gguf(&b, h)?;
            }
        }
        let name = format!("blk.{i}.{}.weight", p.gguf());
        plans.push(TensorPlan {
            name: format!("{name}.lora_a"),
            dims: a.dims().to_vec(),
            dtype: GgmlDType::F32,
        });
        data.push(a);
        plans.push(TensorPlan {
            name: format!("{name}.lora_b"),
            dims: b.dims().to_vec(),
            dtype: GgmlDType::F32,
        });
        data.push(b);
    }
    let md = vec![
        (
            "general.architecture".to_string(),
            Value::String(gguf_arch(model)),
        ),
        ("general.type".to_string(), Value::String("adapter".into())),
        (
            "general.name".to_string(),
            Value::String(format!("{} LoRA", model.cfg.name)),
        ),
        ("adapter.type".to_string(), Value::String("lora".into())),
        (
            "adapter.lora.alpha".to_string(),
            Value::F32(cfg.alpha as f32),
        ),
    ];
    write_gguf(path, &md, &plans, |i| tensor_bytes(&data[i]))?;
    Ok(path.to_path_buf())
}

/// Load an adapter (PEFT directory / adapter_model.safetensors, or llama.cpp .gguf) into the model.
pub fn load_adapter(model: &mut Model, path: &Path) -> Result<String> {
    let is_gguf = path
        .extension()
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false);
    if is_gguf {
        return load_gguf_adapter(model, path);
    }
    let dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent().context("bad adapter path")?.to_path_buf()
    };
    if !dir.join("adapter_model.safetensors").exists() && dir.join("adapter.gguf").exists() {
        return load_gguf_adapter(model, &dir.join("adapter.gguf"));
    }
    let conf: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("adapter_config.json"))
            .context("adapter_config.json not found")?,
    )?;
    let rank = conf["r"]
        .as_u64()
        .context("adapter_config.json: r missing")? as usize;
    let alpha = conf["lora_alpha"].as_f64().unwrap_or(rank as f64);
    let tensors =
        candle_core::safetensors::load(dir.join("adapter_model.safetensors"), &Device::Cpu)?;
    let mut found: HashMap<(usize, Proj), (Option<Tensor>, Option<Tensor>)> = HashMap::new();
    for (k, t) in tensors {
        let Some(rest) = k.split("layers.").nth(1) else {
            continue;
        };
        let Some((idx, tail)) = rest.split_once('.') else {
            continue;
        };
        let Ok(layer) = idx.parse::<usize>() else {
            continue;
        };
        let Some(p) = Proj::ALL
            .into_iter()
            .find(|p| tail.starts_with(&format!("{}.", p.hf())))
        else {
            continue;
        };
        let e = found.entry((layer, p)).or_default();
        if tail.contains("lora_A") {
            e.0 = Some(t);
        } else if tail.contains("lora_B") {
            e.1 = Some(t);
        }
    }
    install_adapter(model, found, alpha, false)?;
    Ok(format!(
        "loaded PEFT adapter (rank {rank}, alpha {alpha}) from {}",
        dir.display()
    ))
}

fn load_gguf_adapter(model: &mut Model, path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let content = gguf_file::Content::read(&mut f)?;
    let md = &content.metadata;
    if md
        .get("general.type")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.as_str())
        != Some("adapter")
    {
        bail!("{} is not a LoRA adapter", path.display());
    }
    let alpha = md
        .get("adapter.lora.alpha")
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(0.0) as f64;
    let mut found: HashMap<(usize, Proj), (Option<Tensor>, Option<Tensor>)> = HashMap::new();
    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    for name in names {
        let Some(rest) = name.strip_prefix("blk.") else {
            continue;
        };
        let Some((idx, tail)) = rest.split_once('.') else {
            continue;
        };
        let Ok(layer) = idx.parse::<usize>() else {
            continue;
        };
        let Some(p) = Proj::ALL
            .into_iter()
            .find(|p| tail.starts_with(&format!("{}.weight.", p.gguf())))
        else {
            continue;
        };
        let t = content
            .tensor(&mut f, &name, &Device::Cpu)?
            .dequantize(&Device::Cpu)?;
        let e = found.entry((layer, p)).or_default();
        if tail.ends_with("lora_a") {
            e.0 = Some(t);
        } else if tail.ends_with("lora_b") {
            e.1 = Some(t);
        }
    }
    let rank = found
        .values()
        .find_map(|(a, _)| a.as_ref().map(|a| a.dim(0).unwrap_or(0)))
        .unwrap_or(0);
    let alpha = if alpha > 0.0 { alpha } else { rank as f64 };
    install_adapter(model, found, alpha, true)?;
    Ok(format!(
        "loaded GGUF adapter (rank {rank}, alpha {alpha}) from {}",
        path.display()
    ))
}

/// `from_gguf` says whether the B matrices are in llama.cpp layout.
fn install_adapter(
    model: &mut Model,
    found: HashMap<(usize, Proj), (Option<Tensor>, Option<Tensor>)>,
    alpha: f64,
    from_gguf: bool,
) -> Result<()> {
    if found.is_empty() {
        bail!("the adapter contains no LoRA weights for this model");
    }
    let src_interleaved = from_gguf && gguf_permutes_qk(model);
    let dst_interleaved = model.cfg.rope_style == RopeStyle::Interleaved;
    let mut targets = vec![];
    let mut rank = 0;
    let mut prepared = vec![];
    for ((layer, p), (a, b)) in found {
        let (Some(a), Some(mut b)) = (a, b) else {
            bail!("adapter is missing A or B for layer {layer} {}", p.short())
        };
        if layer >= model.blocks.len() {
            bail!(
                "adapter has layer {layer} but the model only has {}",
                model.blocks.len()
            );
        }
        if let Some(h) = qk_heads(model, p) {
            if src_interleaved && !dst_interleaved {
                b = rows_gguf_to_hf(&b, h)?;
            } else if !src_interleaved && dst_interleaved {
                b = rows_hf_to_gguf(&b, h)?;
            }
        }
        let lin = model.blocks[layer].proj(p);
        if a.dims() != [a.dim(0)?, lin.in_features()] || b.dims() != [lin.out_features(), a.dim(0)?]
        {
            bail!(
                "adapter shapes for layer {layer} {} do not match this model",
                p.short()
            );
        }
        rank = a.dim(0)?;
        if !targets.contains(&p) {
            targets.push(p);
        }
        prepared.push((layer, p, a, b));
    }
    model.detach_lora();
    let device = model.device.clone();
    for (layer, p, a, b) in prepared {
        let lora = Lora::from_tensors(a.to_device(&device)?, b.to_device(&device)?, alpha)?;
        model.blocks[layer].proj_mut(p).lora = Some(lora);
    }
    targets.sort_by_key(|p| *p as u8);
    model.lora = Some(LoraConfig {
        rank,
        alpha,
        dropout: 0.0,
        targets,
    });
    Ok(())
}

fn lora_delta_cpu(model: &Model, layer: usize, p: Proj) -> Result<Option<Tensor>> {
    match &model.blocks[layer].proj(p).lora {
        Some(l) => Ok(Some(l.delta()?.to_device(&Device::Cpu)?)),
        None => Ok(None),
    }
}

/// Map a GGUF tensor name to (layer, projection) if it is a LoRA-able weight.
fn parse_gguf_weight(name: &str) -> Option<(usize, Proj)> {
    let rest = name.strip_prefix("blk.")?;
    let (idx, tail) = rest.split_once('.')?;
    let layer = idx.parse().ok()?;
    let p = Proj::ALL
        .into_iter()
        .find(|p| tail == format!("{}.weight", p.gguf()))?;
    Some((layer, p))
}

/// Export the base model with the adapter merged in, as a single GGUF file.
pub fn export_merged_gguf(
    model: &Model,
    tok: &Tokenizer,
    out: &Path,
    quant: Option<GgmlDType>,
    progress: &dyn Fn(String),
) -> Result<PathBuf> {
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    match &model.source {
        ModelSource::Gguf { path } => merged_gguf_from_gguf(model, path, out, quant, progress),
        ModelSource::HfDir { path } => merged_gguf_from_hf(model, tok, path, out, quant, progress),
    }
}

fn merged_gguf_from_gguf(
    model: &Model,
    src: &Path,
    out: &Path,
    quant: Option<GgmlDType>,
    progress: &dyn Fn(String),
) -> Result<PathBuf> {
    let mut f = File::open(src)?;
    let content = gguf_file::Content::read(&mut f)?;
    let mut md: Vec<(String, Value)> = content
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    md.sort_by(|a, b| a.0.cmp(&b.0));
    if let Some(q) = quant {
        md.retain(|(k, _)| k != "general.file_type");
        md.push(("general.file_type".into(), Value::U32(file_type_for(q))));
    }
    let mut infos: Vec<(&String, &gguf_file::TensorInfo)> = content.tensor_infos.iter().collect();
    infos.sort_by_key(|(_, i)| i.offset);
    let mut plans = vec![];
    for (name, info) in &infos {
        let dims = info.shape.dims().to_vec();
        let dtype = match quant {
            Some(q) if dims.len() == 2 => fit_dtype(q, dims[1]),
            Some(_)
                if matches!(
                    info.ggml_dtype,
                    GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32
                ) =>
            {
                GgmlDType::F32
            }
            _ => info.ggml_dtype,
        };
        plans.push(TensorPlan {
            name: (*name).clone(),
            dims,
            dtype,
        });
    }
    let total = plans.len();
    write_gguf(out, &md, &plans, |i| {
        let (name, info) = infos[i];
        let plan = &plans[i];
        if i % 16 == 0 {
            progress(format!("writing tensor {}/{}", i + 1, total));
        }
        let delta = match parse_gguf_weight(name) {
            Some((layer, p)) => lora_delta_cpu(model, layer, p)?,
            None => None,
        };
        if delta.is_none() && plan.dtype == info.ggml_dtype {
            let n = tensor_nbytes(info.ggml_dtype, info.shape.elem_count());
            let mut buf = vec![0u8; n];
            f.seek(SeekFrom::Start(content.tensor_data_offset + info.offset))?;
            f.read_exact(&mut buf)?;
            return Ok(buf);
        }
        let w = info
            .read(&mut f, content.tensor_data_offset, &Device::Cpu)?
            .dequantize(&Device::Cpu)?;
        let w = match delta {
            Some(d) => (w + d)?,
            None => w,
        };
        encode_ggml(&w, plan.dtype)
    })?;
    Ok(out.to_path_buf())
}

fn merged_gguf_from_hf(
    model: &Model,
    tok: &Tokenizer,
    dir: &Path,
    out: &Path,
    quant: Option<GgmlDType>,
    progress: &dyn Fn(String),
) -> Result<PathBuf> {
    let c = &model.cfg;
    let arch = gguf_arch(model);
    let q = quant.unwrap_or(GgmlDType::F16);
    let files = hf_weight_files(dir)?;
    // SAFETY: read-only memory map of files that are not modified during export.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&files)? };
    let names: std::collections::HashSet<String> =
        st.tensors().into_iter().map(|(n, _)| n).collect();

    let a = |k: &str| format!("{arch}.{k}");
    let mut md: Vec<(String, Value)> = vec![
        ("general.architecture".into(), Value::String(arch.clone())),
        ("general.name".into(), Value::String(c.name.clone())),
        ("general.file_type".into(), Value::U32(file_type_for(q))),
        ("general.quantization_version".into(), Value::U32(2)),
        (a("context_length"), Value::U32(c.context_length as u32)),
        (a("embedding_length"), Value::U32(c.hidden_size as u32)),
        (a("block_count"), Value::U32(c.num_layers as u32)),
        (
            a("feed_forward_length"),
            Value::U32(c.intermediate_size as u32),
        ),
        (a("attention.head_count"), Value::U32(c.num_heads as u32)),
        (
            a("attention.head_count_kv"),
            Value::U32(c.num_kv_heads as u32),
        ),
        (a("attention.key_length"), Value::U32(c.head_dim as u32)),
        (a("attention.value_length"), Value::U32(c.head_dim as u32)),
        (
            a("attention.layer_norm_rms_epsilon"),
            Value::F32(c.rms_eps as f32),
        ),
        (a("rope.freq_base"), Value::F32(c.rope_theta as f32)),
        (a("rope.dimension_count"), Value::U32(c.head_dim as u32)),
        (a("vocab_size"), Value::U32(c.vocab_size as u32)),
    ];
    if c.rope_linear_scale != 1.0 {
        md.push((a("rope.scaling.type"), Value::String("linear".into())));
        md.push((
            a("rope.scaling.factor"),
            Value::F32(c.rope_linear_scale as f32),
        ));
    }
    // Tokenizer.
    let strings = |v: &[String]| Value::Array(v.iter().map(|s| Value::String(s.clone())).collect());
    let mut tokens = tok.tokens().to_vec();
    tokens.resize(c.vocab_size.max(tokens.len()), String::new());
    for (i, t) in tokens.iter_mut().enumerate() {
        if t.is_empty() {
            *t = format!("[PAD{i}]");
        }
    }
    let mut types = tok.token_types().to_vec();
    types.resize(tokens.len(), crate::tokenizer::TOKEN_UNUSED);
    md.push(("tokenizer.ggml.tokens".into(), strings(&tokens)));
    md.push((
        "tokenizer.ggml.token_type".into(),
        Value::Array(types.iter().map(|t| Value::I32(*t)).collect()),
    ));
    match tok.kind() {
        TokenizerKind::ByteBpe => {
            md.push(("tokenizer.ggml.model".into(), Value::String("gpt2".into())));
            md.push((
                "tokenizer.ggml.pre".into(),
                Value::String(tok.pre_name().to_string()),
            ));
            md.push(("tokenizer.ggml.merges".into(), strings(tok.merges_text())));
        }
        TokenizerKind::SpmScores | TokenizerKind::SpmMerges => {
            let mut scores = vec![0f32; tokens.len()];
            if tok.kind() == TokenizerKind::SpmScores {
                scores[..tok.scores().len()].copy_from_slice(tok.scores());
            } else {
                for (rank, m) in tok.merges_text().iter().enumerate().rev() {
                    if let Some(id) = tok.token_id(&m.replace(' ', "")) {
                        scores[id as usize] = -(rank as f32);
                    }
                }
            }
            md.push(("tokenizer.ggml.model".into(), Value::String("llama".into())));
            md.push((
                "tokenizer.ggml.scores".into(),
                Value::Array(scores.iter().map(|s| Value::F32(*s)).collect()),
            ));
            md.push((
                "tokenizer.ggml.add_space_prefix".into(),
                Value::Bool(tok.add_space_prefix()),
            ));
        }
    }
    for (k, v) in [
        ("tokenizer.ggml.bos_token_id", tok.bos),
        ("tokenizer.ggml.eos_token_id", tok.eos),
        ("tokenizer.ggml.unknown_token_id", tok.unk),
        ("tokenizer.ggml.padding_token_id", tok.pad),
    ] {
        if let Some(id) = v {
            md.push((k.into(), Value::U32(id)));
        }
    }
    md.push((
        "tokenizer.ggml.add_bos_token".into(),
        Value::Bool(tok.add_bos),
    ));
    md.push((
        "tokenizer.ggml.add_eos_token".into(),
        Value::Bool(tok.add_eos),
    ));
    if let Some(t) = &tok.chat_template {
        md.push(("tokenizer.chat_template".into(), Value::String(t.clone())));
    }

    // Tensor plan: (gguf name, hf name, dims, dtype, merge target, permute heads)
    struct Item {
        hf: Option<String>,
        lora: Option<(usize, Proj)>,
        permute: Option<usize>,
        raw: Option<Vec<f32>>,
    }
    let mut plans = vec![];
    let mut items = vec![];
    let mut add = |gguf: String,
                   hf: Option<String>,
                   dims: Vec<usize>,
                   lora: Option<(usize, Proj)>,
                   permute: Option<usize>,
                   raw: Option<Vec<f32>>| {
        let dtype = if dims.len() == 2 {
            fit_dtype(q, dims[1])
        } else {
            GgmlDType::F32
        };
        plans.push(TensorPlan {
            name: gguf,
            dims,
            dtype,
        });
        items.push(Item {
            hf,
            lora,
            permute,
            raw,
        });
    };
    let shape = |n: &str| -> Result<Vec<usize>> { Ok(st.get(n)?.shape().to_vec()) };
    add(
        "token_embd.weight".into(),
        Some("model.embed_tokens.weight".into()),
        shape("model.embed_tokens.weight")?,
        None,
        None,
        None,
    );
    add(
        "output_norm.weight".into(),
        Some("model.norm.weight".into()),
        vec![c.hidden_size],
        None,
        None,
        None,
    );
    if names.contains("lm_head.weight") && !c.tie_embeddings {
        add(
            "output.weight".into(),
            Some("lm_head.weight".into()),
            shape("lm_head.weight")?,
            None,
            None,
            None,
        );
    }
    if let Some(ff) = &c.rope_freq_factors {
        add(
            "rope_freqs.weight".into(),
            None,
            vec![ff.len()],
            None,
            None,
            Some(ff.clone()),
        );
    }
    let permute = arch == "llama";
    for i in 0..c.num_layers {
        let hp = |s: &str| format!("model.layers.{i}.{s}");
        add(
            format!("blk.{i}.attn_norm.weight"),
            Some(hp("input_layernorm.weight")),
            vec![c.hidden_size],
            None,
            None,
            None,
        );
        add(
            format!("blk.{i}.ffn_norm.weight"),
            Some(hp("post_attention_layernorm.weight")),
            vec![c.hidden_size],
            None,
            None,
            None,
        );
        if c.qk_norm {
            add(
                format!("blk.{i}.attn_q_norm.weight"),
                Some(hp("self_attn.q_norm.weight")),
                vec![c.head_dim],
                None,
                None,
                None,
            );
            add(
                format!("blk.{i}.attn_k_norm.weight"),
                Some(hp("self_attn.k_norm.weight")),
                vec![c.head_dim],
                None,
                None,
                None,
            );
        }
        for p in Proj::ALL {
            let heads = if permute { qk_heads(model, p) } else { None };
            let w = hp(&format!("{}.weight", p.hf()));
            add(
                format!("blk.{i}.{}.weight", p.gguf()),
                Some(w.clone()),
                shape(&w)?,
                Some((i, p)),
                heads,
                None,
            );
            let b = hp(&format!("{}.bias", p.hf()));
            if names.contains(&b) {
                add(
                    format!("blk.{i}.{}.bias", p.gguf()),
                    Some(b.clone()),
                    shape(&b)?,
                    None,
                    heads,
                    None,
                );
            }
        }
    }
    let total = plans.len();
    write_gguf(out, &md, &plans, |i| {
        if i % 16 == 0 {
            progress(format!("writing tensor {}/{}", i + 1, total));
        }
        let it = &items[i];
        if let Some(raw) = &it.raw {
            return Ok(raw.iter().flat_map(|x| x.to_le_bytes()).collect());
        }
        let mut w = st
            .load(it.hf.as_ref().unwrap(), &Device::Cpu)?
            .to_dtype(DType::F32)?;
        if let Some((layer, p)) = it.lora {
            if let Some(d) = lora_delta_cpu(model, layer, p)? {
                w = (w + d)?;
            }
        }
        if let Some(h) = it.permute {
            w = if w.rank() == 1 {
                rows_hf_to_gguf(&w.unsqueeze(1)?, h)?.squeeze(1)?
            } else {
                rows_hf_to_gguf(&w, h)?
            };
        }
        encode_ggml(&w, plans[i].dtype)
    })?;
    Ok(out.to_path_buf())
}

/// Export the merged model as Hugging Face safetensors (only for safetensors base models).
pub fn export_merged_safetensors(
    model: &Model,
    out_dir: &Path,
    progress: &dyn Fn(String),
) -> Result<PathBuf> {
    let ModelSource::HfDir { path: src } = &model.source else {
        bail!("merged safetensors export needs a Hugging Face (safetensors) base model; for GGUF models export a merged GGUF instead");
    };
    std::fs::create_dir_all(out_dir)?;
    let files = hf_weight_files(src)?;
    // SAFETY: read-only memory map of files that are not modified during export.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&files)? };
    let mut entries: Vec<(String, String, Vec<usize>, usize)> = st
        .tensors()
        .into_iter()
        .map(|(n, v)| {
            (
                n,
                format!("{:?}", v.dtype()),
                v.shape().to_vec(),
                v.data().len(),
            )
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut header: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
    let mut off = 0usize;
    for (name, dtype, shape, len) in &entries {
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + len]}),
        );
        off += len;
    }
    let mut hjson = serde_json::to_string(&header)?;
    while hjson.len() % 8 != 0 {
        hjson.push(' ');
    }
    let out = out_dir.join("model.safetensors");
    let tmp = out.with_extension("safetensors.part");
    {
        use std::io::Write;
        let mut w = std::io::BufWriter::with_capacity(1 << 22, File::create(&tmp)?);
        w.write_all(&(hjson.len() as u64).to_le_bytes())?;
        w.write_all(hjson.as_bytes())?;
        let total = entries.len();
        for (i, (name, _, _, len)) in entries.iter().enumerate() {
            if i % 16 == 0 {
                progress(format!("writing tensor {}/{}", i + 1, total));
            }
            let target = name
                .strip_prefix("model.layers.")
                .and_then(|r| r.split_once('.'))
                .and_then(|(idx, tail)| {
                    let layer: usize = idx.parse().ok()?;
                    let p = Proj::ALL
                        .into_iter()
                        .find(|p| tail == format!("{}.weight", p.hf()))?;
                    Some((layer, p))
                });
            let delta = match target {
                Some((l, p)) => lora_delta_cpu(model, l, p)?,
                None => None,
            };
            match delta {
                Some(d) => {
                    let base = st.load(name, &Device::Cpu)?;
                    let dt = base.dtype();
                    let merged = (base.to_dtype(DType::F32)? + d)?.to_dtype(dt)?;
                    let bytes = tensor_bytes(&merged)?;
                    if bytes.len() != *len {
                        bail!("size mismatch writing {name}");
                    }
                    w.write_all(&bytes)?;
                }
                None => w.write_all(st.get(name)?.data())?,
            }
        }
        w.flush()?;
    }
    std::fs::rename(&tmp, &out)?;
    for f in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.jinja",
    ] {
        let p = src.join(f);
        if p.exists() {
            std::fs::copy(&p, out_dir.join(f))?;
        }
    }
    Ok(out_dir.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_roundtrip() -> Result<()> {
        let t = Tensor::arange(0f32, 2.0 * 8.0 * 3.0, &Device::Cpu)?.reshape((16, 3))?;
        let g = rows_hf_to_gguf(&t, 2)?;
        let back = rows_gguf_to_hf(&g, 2)?;
        assert_eq!(back.to_vec2::<f32>()?, t.to_vec2::<f32>()?);
        // Row 1 of a GGUF head is HF row d/2 (= 4 for head dim 8).
        assert_eq!(g.get(1)?.to_vec1::<f32>()?, t.get(4)?.to_vec1::<f32>()?);
        Ok(())
    }
}
