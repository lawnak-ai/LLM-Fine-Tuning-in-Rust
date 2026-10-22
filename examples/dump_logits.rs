//! Write the logits of a token sequence to a raw f32 file (for comparing against llama.cpp).
//!
//! cargo run --release --example dump_logits -- <model> <out.bin> <id,id,...> [adapter]

use anyhow::Result;
use candle_core::{Device, Tensor};
use llm_finetune::model::load::{load_model, BasePrecision};
use llm_finetune::model::{Ctx, KvCache};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        anyhow::bail!("usage: dump_logits <model> <out.bin> <id,id,...> [adapter] [precision]");
    }
    let ids: Vec<u32> = args[3]
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let precision = BasePrecision::parse(args.get(5).map(|s| s.as_str()).unwrap_or("auto"))?;
    let (mut model, _tok) = load_model(args[1].as_ref(), &Device::Cpu, precision, &|_| {})?;
    if let Some(a) = args.get(4).filter(|s| !s.is_empty() && *s != "-") {
        eprintln!(
            "{}",
            llm_finetune::export::load_adapter(&mut model, a.as_ref())?
        );
    }
    let input = Tensor::new(ids.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
    let logits = model.forward(&input, &Ctx::inference())?.squeeze(0)?;
    let flat: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    let bytes: Vec<u8> = flat.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(&args[2], bytes)?;

    // The incremental (KV cache) path must agree with the full forward pass.
    let mut cache = KvCache::new(model.blocks.len());
    let split = ids.len() / 2;
    model.forward_cached(&ids[..split], &mut cache)?;
    let mut last = None;
    for &t in &ids[split..] {
        last = Some(model.forward_cached(&[t], &mut cache)?);
    }
    let last: Vec<f32> = last.unwrap().to_vec1()?;
    let full_last = &flat[flat.len() - last.len()..];
    let diff = last
        .iter()
        .zip(full_last)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "wrote {} x {} logits; kv-cache max diff {diff:.2e}",
        ids.len(),
        last.len()
    );
    Ok(())
}
