//! Gradient correctness: autograd vs finite differences, gradient checkpointing vs the
//! full graph, and the QLoRA backward pass vs a dense model.

use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor, TensorId, Var};
use llm_finetune::data::Sample;
use llm_finetune::model::{FrozenWeight, LoraConfig, Model, ModelConfig, Proj, RopeStyle};
use llm_finetune::train::compute_gradients;
use llm_finetune::util::Rng;

fn config(rope_style: RopeStyle, qk_norm: bool, intermediate: usize) -> ModelConfig {
    ModelConfig {
        arch: "llama".into(),
        name: "test".into(),
        vocab_size: 50,
        hidden_size: 32,
        intermediate_size: intermediate,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        rms_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_style,
        rope_linear_scale: 1.0,
        rope_freq_factors: None,
        context_length: 128,
        tie_embeddings: true,
        attn_bias: false,
        qk_norm,
    }
}

fn samples() -> Vec<Sample> {
    vec![
        Sample { ids: vec![1, 5, 9, 13, 2, 7, 33, 41, 8], mask: vec![false, false, false, true, true, true, true, true, true] },
        Sample { ids: vec![3, 4, 10, 49, 12, 6], mask: vec![false, true, true, true, true, true] },
    ]
}

fn model_with_lora(cfg: ModelConfig, dropout: f32) -> Model {
    let mut m = Model::random(cfg, &Device::Cpu, 11).unwrap();
    m.attach_lora(LoraConfig { rank: 4, alpha: 8.0, dropout, targets: Proj::ALL.to_vec() }, 3).unwrap();
    // B starts at zero; make it non-zero so every gradient is informative.
    let mut rng = Rng::new(5);
    for v in m.lora_vars() {
        let n = v.elem_count();
        let vals: Vec<f32> = (0..n).map(|_| rng.normal() * 0.2).collect();
        v.set(&Tensor::from_vec(vals, v.shape(), &Device::Cpu).unwrap()).unwrap();
    }
    m
}

fn loss(m: &Model) -> f64 {
    compute_gradients(m, &samples(), 0, 9, false).unwrap().0
}

fn grad_vec(g: &std::collections::HashMap<TensorId, Tensor>, v: &Var) -> Vec<f32> {
    g.get(&v.id()).expect("missing gradient").flatten_all().unwrap().to_vec1().unwrap()
}

#[test]
fn autograd_matches_finite_differences() {
    for (rope, qk) in [(RopeStyle::Half, false), (RopeStyle::Interleaved, false), (RopeStyle::Half, true)] {
        let m = model_with_lora(config(rope, qk, 48), 0.0);
        let (_, n, grads) = compute_gradients(&m, &samples(), 0, 9, false).unwrap();
        assert_eq!(n, 11);
        let eps = 1e-2f32;
        let mut checked = 0;
        for (vi, v) in m.lora_vars().iter().enumerate().step_by(3) {
            let g = grad_vec(&grads, v);
            let orig: Vec<f32> = v.flatten_all().unwrap().to_vec1().unwrap();
            for idx in [0, orig.len() / 2, orig.len() - 1] {
                let set = |delta: f32| {
                    let mut x = orig.clone();
                    x[idx] += delta;
                    v.set(&Tensor::from_vec(x, v.shape(), &Device::Cpu).unwrap()).unwrap();
                };
                set(eps);
                let lp = loss(&m);
                set(-eps);
                let lm = loss(&m);
                set(0.0);
                let fd = ((lp - lm) / (2.0 * eps as f64)) as f32;
                let tol = 2e-2 * g[idx].abs().max(fd.abs()) + 3e-3;
                assert!((fd - g[idx]).abs() <= tol, "{rope:?} qk_norm={qk} var {vi} idx {idx}: autograd {} vs finite difference {}", g[idx], fd);
                checked += 1;
            }
        }
        assert!(checked > 20);
    }
}

#[test]
fn checkpointing_matches_full_graph_with_dropout() {
    for dropout in [0.0, 0.2] {
        let m = model_with_lora(config(RopeStyle::Interleaved, false, 48), dropout);
        let (l1, _, g1) = compute_gradients(&m, &samples(), 0, 77, false).unwrap();
        let (l2, _, g2) = compute_gradients(&m, &samples(), 0, 77, true).unwrap();
        assert!((l1 - l2).abs() < 1e-4, "loss {l1} vs {l2}");
        for v in m.lora_vars() {
            let (a, b) = (grad_vec(&g1, &v), grad_vec(&g2, &v));
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() <= 1e-4 + 1e-3 * x.abs(), "dropout {dropout}: {x} vs {y}");
            }
        }
    }
}

#[test]
fn quantized_backward_matches_dense() {
    // Feed-forward width 64 so every matrix row is a multiple of the Q8_0 block size.
    let mut q = model_with_lora(config(RopeStyle::Half, false, 64), 0.0);
    let mut d = model_with_lora(config(RopeStyle::Half, false, 64), 0.0);
    for (bq, bd) in q.blocks.iter_mut().zip(d.blocks.iter_mut()) {
        for p in Proj::ALL {
            let w = bq.proj(p).base.weight.to_f32().unwrap();
            let qt = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap();
            bd.proj_mut(p).base.weight = FrozenWeight::Dense(qt.dequantize(&Device::Cpu).unwrap());
            bq.proj_mut(p).base.weight = FrozenWeight::Quant(Arc::new(qt));
        }
    }
    let (lq, _, gq) = compute_gradients(&q, &samples(), 0, 1, true).unwrap();
    let (ld, _, gd) = compute_gradients(&d, &samples(), 0, 1, true).unwrap();
    assert!((lq - ld).abs() / ld.abs() < 1e-2, "loss {lq} vs {ld}");
    let (mut num, mut den) = (0f64, 0f64);
    for v in q.lora_vars().iter().zip(d.lora_vars().iter()) {
        let (a, b) = (grad_vec(&gq, v.0), grad_vec(&gd, v.1));
        for (x, y) in a.iter().zip(b.iter()) {
            num += ((x - y) as f64).powi(2);
            den += (*y as f64).powi(2);
        }
    }
    let rel = (num / den).sqrt();
    assert!(rel < 2e-2, "relative gradient error {rel}");
}
