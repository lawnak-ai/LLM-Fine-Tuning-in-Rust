//! Fused operations with hand-written backward passes.
//!
//! candle's autograd differentiates RMSNorm, RoPE and softmax through a dozen
//! elementwise primitives each. Here the forward pass uses candle's fused
//! kernels (CPU, CUDA and Metal) and the backward pass uses the closed-form
//! gradient. [`ManualGrad`] is the bridge: it copies the already computed
//! output and tells autograd how to map the output gradient to the input.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor, D};

use super::linear::FrozenWeight;

#[derive(Clone)]
pub enum Backward {
    /// y = x · Wᵀ  ⇒  dx = g · W
    Linear(FrozenWeight),
    /// y = rope(x)  ⇒  dx = rope⁻¹(g)
    Rope { cos: Tensor, sin: Tensor, interleaved: bool },
    /// y = x · r · w with r = 1/√(mean(x²)+ε)  ⇒  dx = r·gw − x·r³·mean(gw·x)
    RmsNorm { w: Tensor, eps: f64 },
    /// y = softmax(x)  ⇒  dx = y ⊙ (g − Σ g·y)
    Softmax,
}

pub struct ManualGrad(pub Backward);

fn ensure_fresh(l: &Layout) -> Result<()> {
    if !l.is_contiguous() || l.start_offset() != 0 {
        candle_core::bail!("manual-grad expects a fresh contiguous tensor")
    }
    Ok(())
}

impl CustomOp2 for ManualGrad {
    fn name(&self) -> &'static str {
        "manual-grad"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, Shape)> {
        ensure_fresh(l2)?;
        Ok((s2.try_clone(l2)?, l2.shape().clone()))
    }

    fn cuda_fwd(&self, _: &CudaStorage, _: &Layout, s2: &CudaStorage, l2: &Layout) -> Result<(CudaStorage, Shape)> {
        ensure_fresh(l2)?;
        Ok((s2.try_clone(l2)?, l2.shape().clone()))
    }

    fn metal_fwd(&self, _: &MetalStorage, _: &Layout, s2: &MetalStorage, l2: &Layout) -> Result<(MetalStorage, Shape)> {
        ensure_fresh(l2)?;
        Ok((s2.try_clone(l2)?, l2.shape().clone()))
    }

    fn bwd(&self, x: &Tensor, _y: &Tensor, res: &Tensor, g: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let dx = match &self.0 {
            Backward::Linear(w) => w.backward_input(g)?,
            Backward::Rope { cos, sin, interleaved } => {
                let g = g.contiguous()?;
                let neg = sin.neg()?;
                if *interleaved {
                    candle_nn::rotary_emb::rope_i(&g, cos, &neg)?
                } else {
                    candle_nn::rotary_emb::rope(&g, cos, &neg)?
                }
            }
            Backward::RmsNorm { w, eps } => {
                let x = x.detach();
                let r = (x.sqr()?.mean_keepdim(D::Minus1)? + *eps)?.sqrt()?.recip()?;
                let gw = g.broadcast_mul(w)?;
                let dot = gw.mul(&x)?.mean_keepdim(D::Minus1)?;
                let r3 = r.powf(3.0)?;
                gw.broadcast_mul(&r)?.sub(&x.broadcast_mul(&dot.mul(&r3)?)?)?
            }
            Backward::Softmax => {
                let y = res.detach();
                let s = g.mul(&y)?.sum_keepdim(D::Minus1)?;
                y.mul(&g.broadcast_sub(&s)?)?
            }
        };
        Ok((Some(dx), None))
    }
}

/// Return `y` (computed without autograd) with `x`'s gradient flowing through `b`.
pub fn attach(x: &Tensor, y: Tensor, b: Backward) -> Result<Tensor> {
    if x.track_op() {
        x.apply_op2(&y.contiguous()?, ManualGrad(b))
    } else {
        Ok(y)
    }
}

pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let y = candle_nn::ops::rms_norm(&x.detach().contiguous()?, w, eps as f32)?;
    attach(x, y, Backward::RmsNorm { w: w.clone(), eps })
}

pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let y = candle_nn::ops::softmax_last_dim(&x.detach().contiguous()?)?;
    attach(x, y, Backward::Softmax)
}

/// Rotary embedding on (batch, heads, seq, head_dim); `cos`/`sin` are (seq, head_dim/2).
pub fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor, interleaved: bool) -> Result<Tensor> {
    let xd = x.detach().contiguous()?;
    let (cos, sin) = (cos.contiguous()?, sin.contiguous()?);
    let y = if interleaved {
        candle_nn::rotary_emb::rope_i(&xd, &cos, &sin)?
    } else {
        candle_nn::rotary_emb::rope(&xd, &cos, &sin)?
    };
    attach(x, y, Backward::Rope { cos, sin, interleaved })
}

/// Summed cross-entropy of `logits` (rows, vocab) against `targets` and its gradient
/// with respect to the logits (softmax − one-hot). Multi-threaded on CPU.
pub fn cross_entropy_with_grad(logits: &Tensor, targets: &[u32], want_grad: bool) -> Result<(f64, Option<Tensor>)> {
    let (rows, vocab) = logits.dims2()?;
    let logits = logits.to_dtype(DType::F32)?.contiguous()?;
    if logits.device().is_cpu() {
        use rayon::prelude::*;
        let mut data = logits.flatten_all()?.to_vec1::<f32>()?;
        let nll: f64 = data
            .par_chunks_mut(vocab)
            .zip(targets.par_iter())
            .map(|(row, &t)| {
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                // exp() of very negative numbers is slow (subnormal results) and adds nothing.
                let mut s = 0f64;
                for &v in row.iter() {
                    let d = v - m;
                    if d > -80.0 {
                        s += d.exp() as f64;
                    }
                }
                let lse = m as f64 + s.ln();
                let loss = lse - row[t as usize] as f64;
                if want_grad {
                    // Probabilities below e^-69 (~1e-30) are flushed to zero: as subnormal f32 values
                    // they would make the following matrix product an order of magnitude slower.
                    for v in row.iter_mut() {
                        let d = (*v as f64 - lse) as f32;
                        *v = if d < -69.0 { 0.0 } else { d.exp() };
                    }
                    row[t as usize] -= 1.0;
                }
                loss
            })
            .sum();
        let grad = if want_grad { Some(Tensor::from_vec(data, (rows, vocab), logits.device())?) } else { None };
        return Ok((nll, grad));
    }
    let dev = logits.device();
    let tgt = Tensor::from_slice(targets, (rows, 1), dev)?;
    let m = logits.max_keepdim(D::Minus1)?;
    let e = logits.broadcast_sub(&m)?.exp()?;
    let s = e.sum_keepdim(D::Minus1)?;
    let lse = (s.log()? + &m)?;
    let picked = logits.gather(&tgt, 1)?;
    let nll = lse.sub(&picked)?.sum_all()?.to_scalar::<f32>()? as f64;
    let grad = if want_grad {
        let p = e.broadcast_div(&s)?;
        let minus_one = Tensor::full(-1f32, (rows, 1), dev)?;
        Some(p.scatter_add(&tgt, &minus_one, 1)?)
    } else {
        None
    };
    Ok((nll, grad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    fn rand(shape: &[usize], seed: u64) -> Tensor {
        let mut r = crate::util::Rng::new(seed);
        let n: usize = shape.iter().product();
        Tensor::from_vec((0..n).map(|_| r.normal()).collect::<Vec<f32>>(), shape, &Device::Cpu).unwrap()
    }

    /// Compare a fused op's gradient with autograd through the primitive version.
    fn check(f_fused: impl Fn(&Tensor) -> Tensor, f_ref: impl Fn(&Tensor) -> Tensor, shape: &[usize]) {
        let x = rand(shape, 1);
        let g = rand(shape, 2);
        let a = Var::from_tensor(&x).unwrap();
        let b = Var::from_tensor(&x).unwrap();
        let ya = f_fused(a.as_tensor());
        let yb = f_ref(b.as_tensor());
        let fwd = ya.sub(&yb).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(fwd < 1e-4, "forward differs by {fwd}");
        let ga = ya.mul(&g).unwrap().sum_all().unwrap().backward().unwrap();
        let gb = yb.mul(&g).unwrap().sum_all().unwrap().backward().unwrap();
        let da = ga.get(a.as_tensor()).unwrap();
        let db = gb.get(b.as_tensor()).unwrap();
        let diff = da.sub(db).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-4, "gradient differs by {diff}");
    }

    #[test]
    fn rms_norm_grad() {
        let w = rand(&[8], 3);
        check(
            |x| rms_norm(x, &w, 1e-5).unwrap(),
            |x| {
                let v = x.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
                x.broadcast_div(&(v + 1e-5).unwrap().sqrt().unwrap()).unwrap().broadcast_mul(&w).unwrap()
            },
            &[2, 3, 8],
        );
    }

    #[test]
    fn softmax_grad() {
        check(|x| softmax_last(x).unwrap(), |x| candle_nn::ops::softmax(x, D::Minus1).unwrap(), &[2, 3, 7]);
    }

    #[test]
    fn rope_grad_both_styles() {
        let (t, d) = (5, 8);
        let mut cs = vec![];
        let mut sn = vec![];
        for p in 0..t {
            for i in 0..d / 2 {
                let a = p as f32 * (0.3 + i as f32);
                cs.push(a.cos());
                sn.push(a.sin());
            }
        }
        let cos = Tensor::from_vec(cs, (t, d / 2), &Device::Cpu).unwrap();
        let sin = Tensor::from_vec(sn, (t, d / 2), &Device::Cpu).unwrap();
        for interleaved in [false, true] {
            check(
                |x| rope(x, &cos, &sin, interleaved).unwrap(),
                |x| {
                    if interleaved {
                        candle_nn::rotary_emb::rope_i_slow(x, &cos, &sin).unwrap()
                    } else {
                        candle_nn::rotary_emb::rope_slow(x, &cos, &sin).unwrap()
                    }
                },
                &[1, 2, t, d],
            );
        }
    }

    #[test]
    fn cross_entropy_matches_log_softmax() {
        let logits = rand(&[4, 11], 9);
        let targets = [3u32, 0, 10, 5];
        let (nll, grad) = cross_entropy_with_grad(&logits, &targets, true).unwrap();
        let v = Var::from_tensor(&logits).unwrap();
        let lp = candle_nn::ops::log_softmax(v.as_tensor(), D::Minus1).unwrap();
        let tgt = Tensor::new(&targets, &Device::Cpu).unwrap().unsqueeze(1).unwrap();
        let loss = lp.gather(&tgt, 1).unwrap().neg().unwrap().sum_all().unwrap();
        assert!((loss.to_scalar::<f32>().unwrap() as f64 - nll).abs() < 1e-4);
        let g = loss.backward().unwrap();
        let diff = g.get(v.as_tensor()).unwrap().sub(&grad.unwrap()).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-5, "{diff}");
    }
}
