//! Frozen base weights (dense or quantized) and LoRA adapters.
//!
//! Base weights never get gradients. A plain matmul would make autograd compute
//! a gradient for every frozen weight, so the product is computed outside the
//! graph and re-attached with `ops::attach`, which only passes the gradient on
//! to the input (dx = dy * W). Quantized weights are dequantized on the fly in
//! the backward pass.

use std::sync::Arc;

use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor, Var};

use super::ops::{attach, Backward};
use crate::util::{mix_seed, Rng};

#[derive(Clone, Debug)]
pub enum FrozenWeight {
    /// Dense (out, in) matrix in any float dtype.
    Dense(Tensor),
    /// GGUF quantized (out, in) matrix.
    Quant(Arc<QTensor>),
}

impl FrozenWeight {
    pub fn dims(&self) -> (usize, usize) {
        let d = match self {
            Self::Dense(t) => t.dims().to_vec(),
            Self::Quant(q) => q.shape().dims().to_vec(),
        };
        (d[0], d[1])
    }

    pub fn dtype_name(&self) -> String {
        match self {
            Self::Dense(t) => format!("{:?}", t.dtype()).to_lowercase(),
            Self::Quant(q) => format!("{:?}", q.dtype()).to_lowercase(),
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::Dense(t) => t.elem_count() * t.dtype().size_in_bytes(),
            Self::Quant(q) => q.storage_size_in_bytes(),
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::Dense(t) => t.device().clone(),
            Self::Quant(q) => q.device(),
        }
    }

    /// The weight as a dense f32 tensor on its device.
    pub fn to_f32(&self) -> Result<Tensor> {
        match self {
            Self::Dense(t) => t.to_dtype(DType::F32),
            Self::Quant(q) => q.dequantize(&q.device()),
        }
    }

    /// x * W^T without autograd.
    pub fn forward_t(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(w) => {
                let w = if w.dtype() == x.dtype() {
                    w.clone()
                } else {
                    w.to_dtype(x.dtype())?
                };
                matmul_last(x, &w.t()?)
            }
            Self::Quant(q) => {
                // on cpu dequantize + f32 matmul is faster for big batches,
                // the quantized kernel is only worth it for a few rows
                let rows = x.elem_count() / x.dim(candle_core::D::Minus1)?;
                if q.device().is_cpu() && rows >= 32 {
                    let w = q.dequantize(&q.device())?.to_dtype(x.dtype())?;
                    matmul_last(x, &w.t()?)
                } else {
                    QMatMul::QTensor(q.clone()).forward(&x.contiguous()?)
                }
            }
        }
    }

    /// g * W (gradient w.r.t. the input).
    pub fn backward_input(&self, g: &Tensor) -> Result<Tensor> {
        let w = match self {
            Self::Dense(w) => w.to_dtype(g.dtype())?,
            Self::Quant(q) => q.dequantize(&q.device())?.to_dtype(g.dtype())?,
        };
        matmul_last(g, &w)
    }
}

/// x (..., k) @ w (k, n) as one 2D matmul.
pub fn matmul_last(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let k = dims[dims.len() - 1];
    let n = w.dim(1)?;
    let m: usize = dims[..dims.len() - 1].iter().product();
    let y = x.reshape((m, k))?.matmul(w)?;
    let mut out = dims[..dims.len() - 1].to_vec();
    out.push(n);
    y.reshape(out)
}

/// A linear layer whose weight and bias never train.
#[derive(Clone, Debug)]
pub struct FrozenLinear {
    pub weight: FrozenWeight,
    pub bias: Option<Tensor>,
}

impl FrozenLinear {
    pub fn new(weight: FrozenWeight, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = if x.track_op() {
            let y = self.weight.forward_t(&x.detach())?;
            attach(x, y, Backward::Linear(self.weight.clone()))?
        } else {
            self.weight.forward_t(x)?
        };
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// LoRA adapter, delta W = B * A * (alpha / r).
#[derive(Clone, Debug)]
pub struct Lora {
    /// (r, in)
    pub a: Var,
    /// (out, r)
    pub b: Var,
    pub rank: usize,
    pub alpha: f64,
    pub dropout: f32,
}

impl Lora {
    /// Same init as PEFT: A uniform in +-1/sqrt(in), B zero.
    pub fn new(
        in_f: usize,
        out_f: usize,
        rank: usize,
        alpha: f64,
        dropout: f32,
        device: &Device,
        rng: &mut Rng,
    ) -> Result<Self> {
        let bound = 1.0 / (in_f as f32).sqrt();
        let a: Vec<f32> = (0..rank * in_f)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound)
            .collect();
        let a = Var::from_tensor(&Tensor::from_vec(a, (rank, in_f), device)?)?;
        let b = Var::zeros((out_f, rank), DType::F32, device)?;
        Ok(Self {
            a,
            b,
            rank,
            alpha,
            dropout,
        })
    }

    pub fn from_tensors(a: Tensor, b: Tensor, alpha: f64) -> Result<Self> {
        let rank = a.dim(0)?;
        Ok(Self {
            a: Var::from_tensor(&a.to_dtype(DType::F32)?)?,
            b: Var::from_tensor(&b.to_dtype(DType::F32)?)?,
            rank,
            alpha,
            dropout: 0.0,
        })
    }

    pub fn scale(&self) -> f64 {
        self.alpha / self.rank as f64
    }

    pub fn forward(&self, x: &Tensor, ctx: &Ctx, seed: u64) -> Result<Tensor> {
        let (a, b) = if ctx.grad {
            (self.a.as_tensor().clone(), self.b.as_tensor().clone())
        } else {
            (self.a.as_tensor().detach(), self.b.as_tensor().detach())
        };
        let x = match ctx.dropout_seed {
            Some(s) if self.dropout > 0.0 => dropout(x, self.dropout, mix_seed(&[s, seed]))?,
            _ => x.clone(),
        };
        let h = matmul_last(&x, &a.t()?)?;
        let y = matmul_last(&h, &b.t()?)?;
        y.affine(self.scale(), 0.0)
    }

    /// Dense delta W (out, in) in f32.
    pub fn delta(&self) -> Result<Tensor> {
        self.b
            .as_tensor()
            .matmul(self.a.as_tensor())?
            .affine(self.scale(), 0.0)
    }
}

/// Dropout with a seeded mask so checkpointing can recompute the same one.
fn dropout(x: &Tensor, p: f32, seed: u64) -> Result<Tensor> {
    let n = x.elem_count();
    let mut rng = Rng::new(seed);
    let keep = 1.0 / (1.0 - p);
    let mask: Vec<f32> = (0..n)
        .map(|_| if rng.next_f32() < p { 0.0 } else { keep })
        .collect();
    let mask = Tensor::from_vec(mask, x.shape(), &Device::Cpu)?.to_device(x.device())?;
    x.mul(&mask)
}

/// Options for one forward pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ctx {
    /// Track gradients for LoRA parameters.
    pub grad: bool,
    /// Apply LoRA dropout with this seed (training only).
    pub dropout_seed: Option<u64>,
}

impl Ctx {
    pub fn inference() -> Self {
        Self {
            grad: false,
            dropout_seed: None,
        }
    }
}

/// Base projection + optional adapter.
#[derive(Clone, Debug)]
pub struct Linear {
    pub base: FrozenLinear,
    pub lora: Option<Lora>,
    pub seed: u64,
}

impl Linear {
    pub fn new(base: FrozenLinear, seed: u64) -> Self {
        Self {
            base,
            lora: None,
            seed,
        }
    }

    pub fn in_features(&self) -> usize {
        self.base.weight.dims().1
    }

    pub fn out_features(&self) -> usize {
        self.base.weight.dims().0
    }

    pub fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let y = self.base.forward(x)?;
        match &self.lora {
            Some(l) => y.add(&l.forward(x, ctx, self.seed)?),
            None => Ok(y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grad_route_matches_plain_matmul() -> Result<()> {
        let dev = Device::Cpu;
        let mut rng = Rng::new(1);
        let w: Vec<f32> = (0..6 * 4).map(|_| rng.normal()).collect();
        let w = Tensor::from_vec(w, (6, 4), &dev)?;
        let xv: Vec<f32> = (0..2 * 3 * 4).map(|_| rng.normal()).collect();
        let x = Var::from_tensor(&Tensor::from_vec(xv.clone(), (2, 3, 4), &dev)?)?;
        let lin = FrozenLinear::new(FrozenWeight::Dense(w.clone()), None);
        let y = lin.forward(x.as_tensor())?;
        let loss = y.sqr()?.sum_all()?;
        let grads = loss.backward()?;
        let gx = grads
            .get(x.as_tensor())
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?;

        let x2 = Var::from_tensor(&Tensor::from_vec(xv, (2, 3, 4), &dev)?)?;
        let y2 = x2.as_tensor().broadcast_matmul(&w.t()?)?;
        let loss2 = y2.sqr()?.sum_all()?;
        let grads2 = loss2.backward()?;
        let gx2 = grads2
            .get(x2.as_tensor())
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (a, b) in gx.iter().zip(gx2.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        assert!((loss.to_scalar::<f32>()? - loss2.to_scalar::<f32>()?).abs() < 1e-3);
        Ok(())
    }
}
