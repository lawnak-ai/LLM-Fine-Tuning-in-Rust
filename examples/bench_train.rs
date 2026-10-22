//! Training-speed benchmark on a randomly initialised model with real-world shapes.
//!
//! cargo run --release --example bench_train -- [precision] [seq_len] [steps] [preset]
//!   precision: f32 | f16 | q8_0 | q4_k   (default q8_0)
//!   preset:    qwen0.5b | llama1b | smol135m  (default qwen0.5b)

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device};
use llm_finetune::data::Sample;
use llm_finetune::model::load::BasePrecision;
use llm_finetune::model::{Embedding, FrozenWeight, LoraConfig, Model, ModelConfig, Proj, RopeStyle};
use llm_finetune::train::compute_gradients;

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM")).map(|l| l.to_string()))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn convert(w: &FrozenWeight, p: BasePrecision) -> Result<FrozenWeight> {
    let t = w.to_f32()?;
    Ok(match p {
        BasePrecision::F32 | BasePrecision::Auto => FrozenWeight::Dense(t),
        BasePrecision::F16 => FrozenWeight::Dense(t.to_dtype(DType::F16)?),
        BasePrecision::BF16 => FrozenWeight::Dense(t.to_dtype(DType::BF16)?),
        BasePrecision::Quant(q) => FrozenWeight::Quant(Arc::new(QTensor::quantize(&t, q)?)),
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let precision = BasePrecision::parse(args.get(1).map(|s| s.as_str()).unwrap_or("q8_0"))?;
    let seq: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let preset = args.get(4).map(|s| s.as_str()).unwrap_or("qwen0.5b");
    let (vocab, hidden, inter, layers, heads, kv) = match preset {
        "llama1b" => (128_256, 2048, 8192, 16, 32, 8),
        "smol135m" => (49_152, 576, 1536, 30, 9, 3),
        _ => (151_936, 896, 4864, 24, 14, 2),
    };
    let cfg = ModelConfig {
        arch: "qwen2".into(),
        name: preset.into(),
        vocab_size: vocab,
        hidden_size: hidden,
        intermediate_size: inter,
        num_layers: layers,
        num_heads: heads,
        num_kv_heads: kv,
        head_dim: hidden / heads,
        rms_eps: 1e-6,
        rope_theta: 1e6,
        rope_style: RopeStyle::Half,
        rope_linear_scale: 1.0,
        rope_freq_factors: None,
        context_length: 32768,
        tie_embeddings: true,
        attn_bias: false,
        qk_norm: false,
    };
    let t0 = Instant::now();
    let mut model = Model::random(cfg, &Device::Cpu, 1)?;
    for b in model.blocks.iter_mut() {
        for p in Proj::ALL {
            let w = convert(&b.proj(p).base.weight, precision)?;
            b.proj_mut(p).base.weight = w;
        }
    }
    let head = convert(&model.lm_head.weight, precision)?;
    if let FrozenWeight::Quant(q) = &head {
        model.embed = Embedding::Quant(q.clone());
    }
    model.lm_head.weight = head;
    model.attach_lora(LoraConfig { rank: 16, alpha: 32.0, dropout: 0.0, targets: Proj::ALL.to_vec() }, 1)?;
    eprintln!(
        "{preset}: {} params, weights {:.0} MB ({precision:?}), built in {:.1}s, {} threads",
        model.num_params(),
        model.weights_size_in_bytes() as f64 / 1e6,
        t0.elapsed().as_secs_f64(),
        candle_core::utils::get_num_threads()
    );
    let sample = Sample { ids: (0..seq as u32).map(|i| (i * 7919) % vocab as u32).collect(), mask: vec![true; seq] };
    for ckpt in [true, false] {
        let mut best = f64::MAX;
        for _ in 0..steps {
            let t = Instant::now();
            compute_gradients(&model, std::slice::from_ref(&sample), 0, 1, ckpt)?;
            best = best.min(t.elapsed().as_secs_f64());
        }
        eprintln!(
            "  checkpointing={ckpt:<5}  {:.2}s / step of {seq} tokens  ->  {:.0} tokens/s   (peak RSS {:.0} MB)",
            best,
            seq as f64 / best,
            peak_rss_mb()
        );
    }
    Ok(())
}
