//! End-to-end: data files -> training -> saved adapters -> reload.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};

use candle_core::{Device, Tensor};
use llm_finetune::data::Dataset;
use llm_finetune::export;
use llm_finetune::model::{Ctx, Model, ModelConfig, RopeStyle};
use llm_finetune::tokenizer::Tokenizer;
use llm_finetune::train::{run, TrainConfig, TrainJob, TrainStatus};
use serde_json::json;

/// Byte-level BPE tokenizer with 256 byte tokens, a few merges and an end-of-text token.
fn toy_tokenizer() -> Tokenizer {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).collect();
    bs.extend(0xA1u32..=0xAC);
    bs.extend(0xAEu32..=0xFF);
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut vocab = serde_json::Map::new();
    for (i, c) in cs.iter().enumerate() {
        vocab.insert(char::from_u32(*c).unwrap().to_string(), json!(i));
    }
    let sp = char::from_u32(cs[bs.iter().position(|&b| b == b' ' as u32).unwrap()]).unwrap();
    let merges = vec![format!("{sp} t"), "h e".into(), format!("{sp}t he"), "i s".into()];
    let mut next = 256;
    for m in &merges {
        vocab.insert(m.replace(' ', ""), json!(next));
        next += 1;
    }
    let j = json!({
        "added_tokens": [{"id": next, "content": "<|endoftext|>", "special": true}],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
    });
    let cfg = json!({"eos_token": "<|endoftext|>"});
    Tokenizer::from_hf_json(&j, &cfg, &json!({})).unwrap()
}

fn tiny_model(vocab: usize, rope: RopeStyle) -> Model {
    let cfg = ModelConfig {
        arch: "llama".into(),
        name: "toy".into(),
        vocab_size: vocab,
        hidden_size: 64,
        intermediate_size: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        rms_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_style: rope,
        rope_linear_scale: 1.0,
        rope_freq_factors: None,
        context_length: 512,
        tie_embeddings: true,
        attn_bias: false,
        qk_norm: false,
    };
    Model::random(cfg, &Device::Cpu, 1).unwrap()
}

fn logits(m: &Model, ids: &[u32]) -> Vec<f32> {
    let t = Tensor::new(ids, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
    m.forward(&t, &Ctx::inference()).unwrap().flatten_all().unwrap().to_vec1().unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn train_save_and_reload_adapters() {
    let tok = Arc::new(toy_tokenizer());
    let csv = "question,answer\nWhat is the capital of France?,Paris is the capital of France.\nWhat colour is the sky?,The sky is blue.\nWho wrote Hamlet?,Hamlet was written by Shakespeare.\n";
    let datasets = vec![
        Dataset::from_bytes("d1".into(), "qa.csv", csv.as_bytes()).unwrap(),
        Dataset::from_text("d2".into(), "notes", "Rust is a systems programming language. It is fast and memory safe."),
    ];
    for rope in [RopeStyle::Half, RopeStyle::Interleaved] {
        let model = Arc::new(RwLock::new(tiny_model(tok.vocab_size(), rope)));
        let dir = std::env::temp_dir().join(format!("llmft-test-{}-{:?}", std::process::id(), rope));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = TrainConfig {
            lora_rank: 8,
            lora_alpha: 16.0,
            learning_rate: 5e-3,
            epochs: 0,
            max_steps: 40,
            batch_size: 2,
            grad_accum: 1,
            max_seq_len: 256,
            warmup_ratio: 0.1,
            ..Default::default()
        };
        let status = Arc::new(Mutex::new(TrainStatus::idle()));
        run(TrainJob {
            model: model.clone(),
            tokenizer: tok.clone(),
            datasets: datasets.clone(),
            cfg,
            status: status.clone(),
            stop: Arc::new(AtomicBool::new(false)),
            output_dir: dir.clone(),
        });
        let st = status.lock().unwrap().clone();
        assert_eq!(st.state, "finished", "{}\n{}", st.message, st.logs.join("\n"));
        assert_eq!(st.history.len(), 40);
        assert!(st.logs.iter().any(|l| l.contains("3 conversations -> 3 chat samples")), "{}", st.logs.join("\n"));
        let first: f64 = st.history[..5].iter().map(|p| p.loss).sum::<f64>() / 5.0;
        let last: f64 = st.history[35..].iter().map(|p| p.loss).sum::<f64>() / 5.0;
        assert!(last < first * 0.6, "loss did not go down enough: {first} -> {last}");
        for f in ["adapter_model.safetensors", "adapter_config.json", "adapter.gguf", "training_log.json"] {
            assert!(dir.join(f).exists(), "{f} missing");
        }

        let ids: Vec<u32> = tok.encode("What is the capital of France?", true, false);
        let trained = logits(&model.read().unwrap(), &ids);
        let base = logits(&tiny_model(tok.vocab_size(), rope), &ids);
        assert!(max_diff(&trained, &base) > 1e-2, "adapter has no effect");

        for adapter in [dir.clone(), dir.join("adapter.gguf")] {
            let mut fresh = tiny_model(tok.vocab_size(), rope);
            export::load_adapter(&mut fresh, &adapter).unwrap();
            let d = max_diff(&logits(&fresh, &ids), &trained);
            assert!(d < 1e-4, "{rope:?} reload of {} differs by {d}", adapter.display());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
