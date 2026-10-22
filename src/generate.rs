//! Text generation with a KV cache and the usual sampling controls.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::model::{KvCache, Model};
use crate::tokenizer::Tokenizer;
use crate::util::Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GenParams {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for GenParams {
    fn default() -> Self {
        Self { max_new_tokens: 256, temperature: 0.7, top_p: 0.9, top_k: 40, repeat_penalty: 1.1, repeat_last_n: 64, seed: 0 }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_secs: f64,
    pub gen_secs: f64,
    pub tokens_per_sec: f64,
    pub stop_reason: String,
}

pub fn sample(logits: &mut [f32], p: &GenParams, history: &[u32], rng: &mut Rng) -> u32 {
    if p.repeat_penalty != 1.0 {
        let start = history.len().saturating_sub(p.repeat_last_n);
        let mut seen = history[start..].to_vec();
        seen.sort_unstable();
        seen.dedup();
        for &t in &seen {
            if let Some(l) = logits.get_mut(t as usize) {
                *l = if *l > 0.0 { *l / p.repeat_penalty } else { *l * p.repeat_penalty };
            }
        }
    }
    if p.temperature <= 0.0 {
        return argmax(logits);
    }
    let mut idx: Vec<usize> = (0..logits.len()).filter(|&i| logits[i].is_finite()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    if p.top_k > 0 && idx.len() > p.top_k {
        idx.truncate(p.top_k);
    }
    let max = logits[idx[0]];
    let mut probs: Vec<f32> = idx.iter().map(|&i| ((logits[i] - max) / p.temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for x in probs.iter_mut() {
        *x /= sum;
    }
    if p.top_p > 0.0 && p.top_p < 1.0 {
        let mut acc = 0.0;
        let mut keep = probs.len();
        for (i, x) in probs.iter().enumerate() {
            acc += x;
            if acc >= p.top_p {
                keep = i + 1;
                break;
            }
        }
        probs.truncate(keep);
        idx.truncate(keep);
        let s: f32 = probs.iter().sum();
        for x in probs.iter_mut() {
            *x /= s;
        }
    }
    let r = rng.next_f32();
    let mut acc = 0.0;
    for (i, x) in probs.iter().enumerate() {
        acc += x;
        if r < acc {
            return idx[i] as u32;
        }
    }
    idx[idx.len() - 1] as u32
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Generate from `prompt_ids`. `on_text` receives decoded text increments and may return false to stop.
pub fn generate(
    model: &Model,
    tok: &Tokenizer,
    prompt_ids: &[u32],
    p: &GenParams,
    stop: &AtomicBool,
    mut on_text: impl FnMut(&str) -> bool,
) -> Result<GenStats> {
    let mut stats = GenStats { prompt_tokens: prompt_ids.len(), ..Default::default() };
    let ctx_limit = model.cfg.context_length.max(prompt_ids.len() + 1);
    let eog = tok.end_of_generation_ids();
    let seed = if p.seed == 0 { crate::util::unix_time() ^ 0x5eed } else { p.seed };
    let mut rng = Rng::new(seed);
    let mut cache = KvCache::new(model.blocks.len());
    let t0 = Instant::now();
    let mut logits = model.forward_cached(prompt_ids, &mut cache)?.to_vec1::<f32>()?;
    stats.prompt_secs = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let mut history = prompt_ids.to_vec();
    let mut generated: Vec<u32> = vec![];
    let mut emitted = 0usize;
    stats.stop_reason = "length".into();
    for _ in 0..p.max_new_tokens {
        if stop.load(Ordering::Relaxed) {
            stats.stop_reason = "stopped".into();
            break;
        }
        let next = sample(&mut logits, p, &history, &mut rng);
        if eog.contains(&next) {
            stats.stop_reason = "eos".into();
            break;
        }
        generated.push(next);
        history.push(next);
        // Emit only complete UTF-8 text.
        let bytes = tok.decode_bytes(&generated, true);
        let valid = match std::str::from_utf8(&bytes) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid > emitted {
            let s = std::str::from_utf8(&bytes[emitted..valid]).unwrap_or("");
            emitted = valid;
            if !on_text(s) {
                stats.stop_reason = "stopped".into();
                break;
            }
        }
        if history.len() >= ctx_limit {
            stats.stop_reason = "context".into();
            break;
        }
        logits = model.forward_cached(&[next], &mut cache)?.to_vec1::<f32>()?;
    }
    stats.generated_tokens = generated.len();
    stats.gen_secs = t1.elapsed().as_secs_f64();
    stats.tokens_per_sec = generated.len() as f64 / stats.gen_secs.max(1e-9);
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_top_k() {
        let p = GenParams { temperature: 0.0, repeat_penalty: 1.0, ..Default::default() };
        let mut rng = Rng::new(1);
        assert_eq!(sample(&mut [0.1, 2.0, 0.5], &p, &[], &mut rng), 1);
        let p = GenParams { temperature: 1.0, top_k: 1, repeat_penalty: 1.0, ..Default::default() };
        for _ in 0..20 {
            assert_eq!(sample(&mut [0.1, 2.0, 1.9], &p, &[], &mut rng), 1);
        }
        // Repetition penalty can flip the choice.
        let p = GenParams { temperature: 0.0, repeat_penalty: 2.0, ..Default::default() };
        assert_eq!(sample(&mut [0.0, 2.0, 1.5], &p, &[1], &mut rng), 2);
    }
}
