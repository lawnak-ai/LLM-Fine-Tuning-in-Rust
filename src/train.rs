//! LoRA / QLoRA training loop.
//!
//! Memory savers:
//! * frozen weights get no gradients (see `model::linear`),
//! * optional gradient checkpointing: blocks are re-run one at a time during
//!   the backward pass so only one block's activations live at once,
//! * the vocabulary projection + cross-entropy run in chunks, so the
//!   (tokens x vocab) logits matrix is never materialised in full.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor, TensorId, Var};
use serde::{Deserialize, Serialize};

use crate::chat::{ChatFormatter, TemplateKind};
use crate::data::{build_samples, split_eval, DataMode, Dataset, Sample, SampleOptions};
use crate::export;
use crate::model::ops::cross_entropy_with_grad;
use crate::model::{Ctx, LoraConfig, Model, Proj};
use crate::tokenizer::Tokenizer;
use crate::util::{mix_seed, Rng};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainConfig {
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub lora_dropout: f32,
    pub target_modules: Vec<String>,
    pub learning_rate: f64,
    pub epochs: usize,
    /// 0 = derived from epochs.
    pub max_steps: usize,
    pub batch_size: usize,
    pub grad_accum: usize,
    pub max_seq_len: usize,
    pub warmup_ratio: f64,
    /// "cosine", "linear" or "constant".
    pub lr_scheduler: String,
    pub weight_decay: f64,
    pub max_grad_norm: f64,
    pub seed: u64,
    pub gradient_checkpointing: bool,
    pub eval_fraction: f64,
    /// Evaluate every N steps (0 = only at the end).
    pub eval_every: usize,
    /// Save a checkpoint every N steps (0 = only at the end).
    pub save_every: usize,
    pub output_dir: String,
    pub train_on_prompt: bool,
    pub chat_template: String,
    pub system_prompt: String,
    /// "auto" or "text".
    pub data_mode: String,
    /// Continue training the adapter that is currently loaded instead of starting fresh.
    pub resume_adapter: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lora_rank: 16,
            lora_alpha: 32.0,
            lora_dropout: 0.0,
            target_modules: Proj::ALL.iter().map(|p| p.short().to_string()).collect(),
            learning_rate: 2e-4,
            epochs: 3,
            max_steps: 0,
            batch_size: 1,
            grad_accum: 4,
            max_seq_len: 512,
            warmup_ratio: 0.05,
            lr_scheduler: "cosine".into(),
            weight_decay: 0.0,
            max_grad_norm: 1.0,
            seed: 42,
            gradient_checkpointing: true,
            eval_fraction: 0.0,
            eval_every: 0,
            save_every: 0,
            output_dir: String::new(),
            train_on_prompt: false,
            chat_template: "auto".into(),
            system_prompt: String::new(),
            data_mode: "auto".into(),
            resume_adapter: false,
        }
    }
}

impl TrainConfig {
    pub fn targets(&self) -> Result<Vec<Proj>> {
        let mut v = vec![];
        for t in &self.target_modules {
            match t.as_str() {
                "all" | "all-linear" => return Ok(Proj::ALL.to_vec()),
                "attn" | "attention" => v.extend([Proj::Q, Proj::K, Proj::V, Proj::O]),
                "mlp" | "ffn" => v.extend([Proj::Gate, Proj::Up, Proj::Down]),
                other => v.push(Proj::parse(other).with_context(|| format!("unknown target module '{other}'"))?),
            }
        }
        v.sort_by_key(|p| *p as u8);
        v.dedup();
        Ok(v)
    }

    fn validate(&self) -> Result<()> {
        if self.batch_size == 0 || self.grad_accum == 0 {
            bail!("batch size and gradient accumulation must be at least 1");
        }
        if !(self.learning_rate > 0.0 && self.learning_rate < 1.0) {
            bail!("learning rate must be between 0 and 1");
        }
        if !(0.0..1.0).contains(&self.lora_dropout) {
            bail!("LoRA dropout must be in [0, 1)");
        }
        if self.epochs == 0 && self.max_steps == 0 {
            bail!("set epochs or max steps");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct StepPoint {
    pub step: usize,
    pub loss: f64,
    pub lr: f64,
    pub grad_norm: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct TrainStatus {
    /// idle | preparing | training | stopping | finished | stopped | failed
    pub state: String,
    pub message: String,
    pub step: usize,
    pub total_steps: usize,
    pub epoch: f64,
    pub loss: f64,
    pub smoothed_loss: f64,
    pub lr: f64,
    pub grad_norm: f64,
    pub tokens_per_sec: f64,
    pub elapsed_secs: f64,
    pub eta_secs: f64,
    pub trained_tokens: u64,
    pub train_samples: usize,
    pub eval_samples: usize,
    pub eval_loss: Option<f64>,
    pub trainable_params: usize,
    pub chat_template: String,
    pub history: Vec<StepPoint>,
    pub eval_history: Vec<(usize, f64)>,
    pub logs: Vec<String>,
    pub output_dir: Option<String>,
    pub saved_files: Vec<String>,
}

impl TrainStatus {
    pub fn idle() -> Self {
        Self { state: "idle".into(), ..Default::default() }
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        eprintln!("[train] {msg}");
        self.logs.push(msg);
        if self.logs.len() > 1000 {
            self.logs.drain(0..200);
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self.state.as_str(), "preparing" | "training" | "stopping")
    }
}

/// A padded batch ready for the model.
struct Batch {
    /// (batch, seq) input ids
    input: Tensor,
    /// flat positions (into batch*seq) that have a target
    positions: Tensor,
    targets: Tensor,
    n_targets: usize,
    n_tokens: usize,
}

fn make_batch(samples: &[&Sample], pad: u32, device: &Device) -> Result<Batch> {
    let t = samples.iter().map(|s| s.ids.len()).max().unwrap_or(1).saturating_sub(1).max(1);
    let b = samples.len();
    let mut input = vec![pad; b * t];
    let mut positions = vec![];
    let mut targets = vec![];
    let mut n_tokens = 0;
    for (bi, s) in samples.iter().enumerate() {
        let n = s.ids.len().saturating_sub(1);
        for i in 0..n {
            input[bi * t + i] = s.ids[i];
            if s.mask[i + 1] {
                positions.push((bi * t + i) as u32);
                targets.push(s.ids[i + 1]);
            }
        }
        n_tokens += n;
    }
    let n_targets = targets.len();
    Ok(Batch {
        input: Tensor::from_vec(input, (b, t), device)?,
        positions: Tensor::from_vec(positions, n_targets, device)?,
        targets: Tensor::from_vec(targets, n_targets, device)?,
        n_targets,
        n_tokens,
    })
}

/// Rows of the vocabulary projection processed at once.
fn head_chunk(vocab: usize) -> usize {
    ((1usize << 25) / vocab.max(1)).clamp(8, 4096)
}

/// Final norm + LM head + summed cross-entropy. Returns (loss sum, dL/dh).
///
/// The vocabulary projection is the largest matrix of the model, so it runs in row
/// chunks with a fused cross-entropy and a manual backward pass (no autograd graph).
fn head_loss_and_grad(model: &Model, h: &Tensor, batch: &Batch) -> Result<(f64, Tensor)> {
    let (b, t, d) = h.dims3()?;
    let hv = Var::from_tensor(h)?;
    let hn = model.final_norm(hv.as_tensor())?;
    let sel = hn.reshape((b * t, d))?.index_select(&batch.positions, 0)?;
    let sel_det = sel.detach();
    let chunk = head_chunk(model.cfg.vocab_size);
    let targets = batch.targets.to_vec1::<u32>()?;
    let mut grads = vec![];
    let mut loss_sum = 0.0;
    let mut start = 0;
    while start < batch.n_targets {
        let c = chunk.min(batch.n_targets - start);
        let logits = model.logits(&sel_det.narrow(0, start, c)?)?;
        let (nll, g) = cross_entropy_with_grad(&logits, &targets[start..start + c], true)?;
        loss_sum += nll;
        grads.push(model.lm_head.weight.backward_input(&g.context("missing logits gradient")?)?);
        start += c;
    }
    let gsel = Tensor::cat(&grads, 0)?;
    let gs = sel.mul(&gsel)?.sum_all()?.backward()?;
    let gh = gs.get(hv.as_tensor()).context("missing hidden gradient")?.clone();
    Ok((loss_sum, gh))
}

/// Forward + backward for one micro batch. Gradients are for the *summed* loss.
fn micro_step(model: &Model, batch: &Batch, seed: u64, checkpointing: bool) -> Result<(f64, HashMap<TensorId, Tensor>)> {
    let n_layers = model.blocks.len();
    let ctx_ng = Ctx { grad: false, dropout_seed: Some(seed) };
    let ctx_g = Ctx { grad: true, dropout_seed: Some(seed) };
    let x0 = model.embed(&batch.input)?;
    let mut out: HashMap<TensorId, Tensor> = HashMap::new();
    if batch.n_targets == 0 {
        return Ok((0.0, out));
    }
    if checkpointing {
        let mut inputs = Vec::with_capacity(n_layers);
        let mut x = x0;
        for li in 0..n_layers {
            inputs.push(x.clone());
            x = model.block_forward(li, &x, &ctx_ng, None)?;
        }
        let (loss, mut g) = head_loss_and_grad(model, &x, batch)?;
        drop(x);
        for li in (0..n_layers).rev() {
            let xin = inputs.pop().unwrap();
            let vars = model.layer_lora_vars(li);
            let xv = if li > 0 { Some(Var::from_tensor(&xin)?) } else { None };
            let x_in = xv.as_ref().map(|v| v.as_tensor().clone()).unwrap_or(xin);
            let y = model.block_forward(li, &x_in, &ctx_g, None)?;
            let gs = y.mul(&g)?.sum_all()?.backward()?;
            for v in vars {
                if let Some(gv) = gs.get(v.as_tensor()) {
                    out.insert(v.id(), gv.clone());
                }
            }
            if let Some(xv) = xv {
                g = gs.get(xv.as_tensor()).context("missing block input gradient")?.clone();
            }
        }
        Ok((loss, out))
    } else {
        let mut x = x0;
        for li in 0..n_layers {
            x = model.block_forward(li, &x, &ctx_g, None)?;
        }
        let (loss, g) = head_loss_and_grad(model, &x.detach(), batch)?;
        let gs = x.mul(&g)?.sum_all()?.backward()?;
        for v in model.lora_vars() {
            if let Some(gv) = gs.get(v.as_tensor()) {
                out.insert(v.id(), gv.clone());
            }
        }
        Ok((loss, out))
    }
}

/// Summed loss, number of targets and LoRA gradients for one batch (used by tests and tools).
pub fn compute_gradients(
    model: &Model,
    samples: &[Sample],
    pad: u32,
    seed: u64,
    checkpointing: bool,
) -> Result<(f64, usize, HashMap<TensorId, Tensor>)> {
    let refs: Vec<&Sample> = samples.iter().collect();
    let batch = make_batch(&refs, pad, &model.device)?;
    let (loss, grads) = micro_step(model, &batch, seed, checkpointing)?;
    Ok((loss, batch.n_targets, grads))
}

/// Mean loss over samples without gradients.
pub fn evaluate(model: &Model, samples: &[Sample], batch_size: usize, pad: u32) -> Result<f64> {
    let mut sum = 0.0;
    let mut n = 0usize;
    for chunk in samples.chunks(batch_size.max(1)) {
        let refs: Vec<&Sample> = chunk.iter().collect();
        let batch = make_batch(&refs, pad, &model.device)?;
        if batch.n_targets == 0 {
            continue;
        }
        let mut x = model.embed(&batch.input)?;
        for li in 0..model.blocks.len() {
            x = model.block_forward(li, &x, &Ctx::inference(), None)?;
        }
        let (b, t, d) = x.dims3()?;
        let hn = model.final_norm(&x)?.reshape((b * t, d))?.index_select(&batch.positions, 0)?;
        let chunk = head_chunk(model.cfg.vocab_size);
        let targets = batch.targets.to_vec1::<u32>()?;
        let mut start = 0;
        while start < batch.n_targets {
            let c = chunk.min(batch.n_targets - start);
            let logits = model.logits(&hn.narrow(0, start, c)?)?;
            sum += cross_entropy_with_grad(&logits, &targets[start..start + c], false)?.0;
            start += c;
        }
        n += batch.n_targets;
    }
    Ok(if n > 0 { sum / n as f64 } else { f64::NAN })
}

struct AdamW {
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    t: i32,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

impl AdamW {
    fn new(vars: &[Var], weight_decay: f64) -> Result<Self> {
        let zeros = |v: &Var| v.as_tensor().zeros_like();
        Ok(Self {
            m: vars.iter().map(zeros).collect::<candle_core::Result<_>>()?,
            v: vars.iter().map(zeros).collect::<candle_core::Result<_>>()?,
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
        })
    }

    fn step(&mut self, vars: &[Var], grads: &[Tensor], lr: f64) -> Result<()> {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t);
        let bc2 = 1.0 - self.beta2.powi(self.t);
        for (i, var) in vars.iter().enumerate() {
            let g = &grads[i];
            let m = (self.m[i].affine(self.beta1, 0.0)? + g.affine(1.0 - self.beta1, 0.0)?)?;
            let v = (self.v[i].affine(self.beta2, 0.0)? + g.sqr()?.affine(1.0 - self.beta2, 0.0)?)?;
            let update = m.affine(1.0 / bc1, 0.0)?.div(&(v.affine(1.0 / bc2, 0.0)?.sqrt()? + self.eps)?)?;
            let p = var.as_tensor().detach();
            let p = if self.weight_decay > 0.0 { p.affine(1.0 - lr * self.weight_decay, 0.0)? } else { p };
            var.set(&(p - update.affine(lr, 0.0)?)?)?;
            self.m[i] = m;
            self.v[i] = v;
        }
        Ok(())
    }
}

fn lr_at(cfg: &TrainConfig, step: usize, total: usize) -> f64 {
    let warmup = ((total as f64) * cfg.warmup_ratio).ceil() as usize;
    if step < warmup {
        return cfg.learning_rate * (step + 1) as f64 / warmup as f64;
    }
    let progress = if total > warmup { (step - warmup) as f64 / (total - warmup) as f64 } else { 1.0 };
    match cfg.lr_scheduler.as_str() {
        "constant" => cfg.learning_rate,
        "linear" => cfg.learning_rate * (1.0 - progress).max(0.0),
        _ => cfg.learning_rate * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()),
    }
}

pub struct TrainJob {
    pub model: Arc<RwLock<Model>>,
    pub tokenizer: Arc<Tokenizer>,
    pub datasets: Vec<Dataset>,
    pub cfg: TrainConfig,
    pub status: Arc<Mutex<TrainStatus>>,
    pub stop: Arc<AtomicBool>,
    pub output_dir: PathBuf,
}

/// Run a job to completion and record the outcome in its status.
pub fn run(job: TrainJob) {
    let status = job.status.clone();
    let result = run_inner(&job);
    let mut st = status.lock().unwrap();
    match result {
        Ok(()) => {}
        Err(e) => {
            st.state = "failed".into();
            st.message = format!("{e:#}");
            let m = st.message.clone();
            st.log(format!("error: {m}"));
        }
    }
}

fn set_state(status: &Mutex<TrainStatus>, state: &str, msg: &str) {
    let mut st = status.lock().unwrap();
    st.state = state.into();
    st.message = msg.into();
    st.log(msg);
}

fn run_inner(job: &TrainJob) -> Result<()> {
    let cfg = &job.cfg;
    cfg.validate()?;
    let status = &job.status;
    set_state(status, "preparing", "Preparing data…");

    let tok = &job.tokenizer;
    let template = TemplateKind::parse(&cfg.chat_template)?;
    let formatter = ChatFormatter::new(template, tok);
    let mode = if cfg.data_mode == "text" { DataMode::Text } else { DataMode::Auto };
    let system = if cfg.system_prompt.trim().is_empty() { None } else { Some(cfg.system_prompt.clone()) };
    let opts = SampleOptions {
        max_len: cfg.max_seq_len,
        train_on_prompt: cfg.train_on_prompt,
        system_prompt: system,
        mode,
        formatter: &formatter,
    };
    let (samples, stats) = build_samples(&job.datasets, tok, &opts)?;
    let (train, eval) = split_eval(samples, cfg.eval_fraction, cfg.seed);
    {
        let mut st = status.lock().unwrap();
        st.chat_template = formatter.name();
        st.train_samples = train.len();
        st.eval_samples = eval.len();
        st.log(format!(
            "data: {} documents -> {} text chunks, {} conversations -> {} chat samples (template: {}), {} truncated, {} dropped; {} tokens, {} trained",
            stats.documents,
            stats.text_chunks,
            stats.conversations,
            stats.chat_samples,
            formatter.name(),
            stats.truncated,
            stats.dropped,
            stats.total_tokens,
            stats.target_tokens
        ));
        if !eval.is_empty() {
            st.log(format!("{} samples held out for evaluation", eval.len()));
        }
    }

    let targets = cfg.targets()?;
    {
        let mut model = job.model.write().unwrap();
        let reuse = cfg.resume_adapter && model.lora.as_ref().map(|l| l.targets == targets).unwrap_or(false);
        if reuse {
            model.set_lora_dropout(cfg.lora_dropout);
            status.lock().unwrap().log("continuing with the loaded adapter");
        } else {
            let lc = LoraConfig { rank: cfg.lora_rank, alpha: cfg.lora_alpha, dropout: cfg.lora_dropout, targets: targets.clone() };
            model.attach_lora(lc, cfg.seed)?;
        }
    }
    let model = job.model.read().unwrap();
    let vars = model.lora_vars();
    let trainable: usize = vars.iter().map(|v| v.elem_count()).sum();
    let pad = tok.pad_id();

    let micro_per_epoch = train.len().div_ceil(cfg.batch_size);
    let steps_per_epoch = micro_per_epoch.div_ceil(cfg.grad_accum).max(1);
    let total_steps = if cfg.max_steps > 0 { cfg.max_steps } else { steps_per_epoch * cfg.epochs };
    {
        let mut st = status.lock().unwrap();
        st.state = "training".into();
        st.total_steps = total_steps;
        st.trainable_params = trainable;
        st.output_dir = Some(job.output_dir.display().to_string());
        st.message = "Training…".into();
        st.log(format!(
            "training {} LoRA parameters ({:.3}% of the model) for {} steps on {} (rank {}, alpha {}, batch {} x {} accum, seq {})",
            trainable,
            100.0 * trainable as f64 / model.num_params() as f64,
            total_steps,
            crate::device::device_label(&model.device),
            cfg.lora_rank,
            cfg.lora_alpha,
            cfg.batch_size,
            cfg.grad_accum,
            cfg.max_seq_len
        ));
    }

    let mut opt = AdamW::new(&vars, cfg.weight_decay)?;
    let mut rng = Rng::new(cfg.seed);
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut cursor = order.len();
    let mut epoch = 0usize;
    let started = Instant::now();
    let mut trained_tokens: u64 = 0;
    let mut smoothed = f64::NAN;
    let mut stopped = false;
    std::fs::create_dir_all(&job.output_dir)?;

    for step in 0..total_steps {
        let lr = lr_at(cfg, step, total_steps);
        let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
        let mut loss_sum = 0.0;
        let mut n_targets = 0usize;
        for micro in 0..cfg.grad_accum {
            if job.stop.load(Ordering::Relaxed) {
                stopped = true;
                break;
            }
            if cursor >= order.len() {
                rng.shuffle(&mut order);
                cursor = 0;
                epoch += 1;
            }
            let end = (cursor + cfg.batch_size).min(order.len());
            let refs: Vec<&Sample> = order[cursor..end].iter().map(|&i| &train[i]).collect();
            cursor = end;
            let batch = make_batch(&refs, pad, &model.device)?;
            let seed = mix_seed(&[cfg.seed, step as u64, micro as u64]);
            let (l, grads) = micro_step(&model, &batch, seed, cfg.gradient_checkpointing)?;
            loss_sum += l;
            n_targets += batch.n_targets;
            trained_tokens += batch.n_tokens as u64;
            for (i, v) in vars.iter().enumerate() {
                if let Some(g) = grads.get(&v.id()) {
                    acc[i] = Some(match acc[i].take() {
                        Some(a) => (a + g)?,
                        None => g.clone(),
                    });
                }
            }
        }
        if stopped {
            break;
        }
        if n_targets == 0 {
            continue;
        }
        // Mean over all target tokens of the accumulated micro batches.
        let scale = 1.0 / n_targets as f64;
        let mut grads = Vec::with_capacity(vars.len());
        let mut sq = 0.0f64;
        for (i, v) in vars.iter().enumerate() {
            let g = match acc[i].take() {
                Some(g) => g.affine(scale, 0.0)?,
                None => v.as_tensor().zeros_like()?,
            };
            sq += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            grads.push(g);
        }
        let norm = sq.sqrt();
        if !norm.is_finite() {
            bail!("gradient became NaN/inf at step {} — try a lower learning rate", step + 1);
        }
        if cfg.max_grad_norm > 0.0 && norm > cfg.max_grad_norm {
            let s = cfg.max_grad_norm / (norm + 1e-6);
            for g in grads.iter_mut() {
                *g = g.affine(s, 0.0)?;
            }
        }
        opt.step(&vars, &grads, lr)?;

        let loss = loss_sum / n_targets as f64;
        smoothed = if smoothed.is_nan() { loss } else { 0.9 * smoothed + 0.1 * loss };
        let elapsed = started.elapsed().as_secs_f64();
        {
            let mut st = status.lock().unwrap();
            st.step = step + 1;
            st.loss = loss;
            st.smoothed_loss = smoothed;
            st.lr = lr;
            st.grad_norm = norm;
            st.epoch = (epoch.max(1) - 1) as f64 + cursor as f64 / order.len().max(1) as f64;
            st.elapsed_secs = elapsed;
            st.trained_tokens = trained_tokens;
            st.tokens_per_sec = trained_tokens as f64 / elapsed.max(1e-6);
            st.eta_secs = elapsed / (step + 1) as f64 * (total_steps - step - 1) as f64;
            st.history.push(StepPoint { step: step + 1, loss, lr, grad_norm: norm });
            let tps = st.tokens_per_sec;
            st.log(format!("step {}/{}  loss {:.4}  lr {:.2e}  grad-norm {:.3}  {:.0} tok/s", step + 1, total_steps, loss, lr, norm, tps));
        }
        let last = step + 1 == total_steps;
        if !eval.is_empty() && ((cfg.eval_every > 0 && (step + 1) % cfg.eval_every == 0) || last) {
            let el = evaluate(&model, &eval, cfg.batch_size, pad)?;
            let mut st = status.lock().unwrap();
            st.eval_loss = Some(el);
            st.eval_history.push((step + 1, el));
            st.log(format!("eval loss {:.4} (perplexity {:.2})", el, el.exp()));
        }
        if cfg.save_every > 0 && (step + 1) % cfg.save_every == 0 && !last {
            let dir = job.output_dir.join(format!("checkpoint-{}", step + 1));
            let files = export::save_peft_adapter(&model, tok, &dir)?;
            status.lock().unwrap().log(format!("checkpoint saved to {} ({} files)", dir.display(), files.len()));
        }
    }

    let files = save_outputs(&model, tok, job, cfg)?;
    let mut st = status.lock().unwrap();
    st.saved_files = files.iter().map(|p| p.display().to_string()).collect();
    if stopped {
        st.state = "stopped".into();
        st.message = format!("Stopped at step {}. Adapter saved.", st.step);
    } else {
        st.state = "finished".into();
        st.message = "Training finished. Adapter saved.".into();
    }
    let msg = st.message.clone();
    st.log(format!("{msg} Output: {}", job.output_dir.display()));
    Ok(())
}

fn save_outputs(model: &Model, tok: &Tokenizer, job: &TrainJob, cfg: &TrainConfig) -> Result<Vec<PathBuf>> {
    let dir = &job.output_dir;
    let mut files = export::save_peft_adapter(model, tok, dir)?;
    match export::save_gguf_adapter(model, &dir.join("adapter.gguf")) {
        Ok(p) => files.push(p),
        Err(e) => job.status.lock().unwrap().log(format!("could not write GGUF adapter: {e:#}")),
    }
    let st = job.status.lock().unwrap().clone();
    let log = serde_json::json!({
        "config": cfg,
        "history": st.history,
        "eval_history": st.eval_history,
        "trained_tokens": st.trained_tokens,
        "elapsed_secs": st.elapsed_secs,
    });
    let p = dir.join("training_log.json");
    std::fs::write(&p, serde_json::to_string_pretty(&log)?)?;
    files.push(p);
    Ok(files)
}
