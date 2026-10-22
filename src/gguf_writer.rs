//! GGUF v3 writer. Tensors are written one at a time so a merged export doesn't
//! need a second copy of the model in memory.

use std::io::{Seek, Write};
use std::path::Path;

use anyhow::{bail, Result};
use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::GgmlDType;

pub fn ggml_type_id(t: GgmlDType) -> u32 {
    match t {
        GgmlDType::F32 => 0,
        GgmlDType::F16 => 1,
        GgmlDType::Q4_0 => 2,
        GgmlDType::Q4_1 => 3,
        GgmlDType::Q5_0 => 6,
        GgmlDType::Q5_1 => 7,
        GgmlDType::Q8_0 => 8,
        GgmlDType::Q8_1 => 9,
        GgmlDType::Q2K => 10,
        GgmlDType::Q3K => 11,
        GgmlDType::Q4K => 12,
        GgmlDType::Q5K => 13,
        GgmlDType::Q6K => 14,
        GgmlDType::Q8K => 15,
        GgmlDType::BF16 => 30,
    }
}

/// llama.cpp `general.file_type` for a uniform quantization.
pub fn file_type_for(t: GgmlDType) -> u32 {
    match t {
        GgmlDType::F32 => 0,
        GgmlDType::F16 => 1,
        GgmlDType::Q4_0 => 2,
        GgmlDType::Q4_1 => 3,
        GgmlDType::Q8_0 => 7,
        GgmlDType::Q5_0 => 8,
        GgmlDType::Q5_1 => 9,
        GgmlDType::Q2K => 10,
        GgmlDType::Q3K => 12,
        GgmlDType::Q4K => 15,
        GgmlDType::Q5K => 17,
        GgmlDType::Q6K => 18,
        GgmlDType::BF16 => 32,
        GgmlDType::Q8_1 | GgmlDType::Q8K => 7,
    }
}

pub fn tensor_nbytes(dtype: GgmlDType, elems: usize) -> usize {
    elems / dtype.block_size() * dtype.type_size()
}

/// Description of one tensor in the file. `dims` are in row-major (candle) order.
#[derive(Debug, Clone)]
pub struct TensorPlan {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: GgmlDType,
}

impl TensorPlan {
    pub fn nbytes(&self) -> usize {
        tensor_nbytes(self.dtype, self.dims.iter().product())
    }
}

fn value_type_id(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn write_str<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn write_value<W: Write>(w: &mut W, v: &Value) -> Result<()> {
    match v {
        Value::U8(x) => w.write_all(&x.to_le_bytes())?,
        Value::I8(x) => w.write_all(&x.to_le_bytes())?,
        Value::U16(x) => w.write_all(&x.to_le_bytes())?,
        Value::I16(x) => w.write_all(&x.to_le_bytes())?,
        Value::U32(x) => w.write_all(&x.to_le_bytes())?,
        Value::I32(x) => w.write_all(&x.to_le_bytes())?,
        Value::U64(x) => w.write_all(&x.to_le_bytes())?,
        Value::I64(x) => w.write_all(&x.to_le_bytes())?,
        Value::F32(x) => w.write_all(&x.to_le_bytes())?,
        Value::F64(x) => w.write_all(&x.to_le_bytes())?,
        Value::Bool(x) => w.write_all(&[*x as u8])?,
        Value::String(s) => write_str(w, s)?,
        Value::Array(items) => {
            let ty = items.first().map(value_type_id).unwrap_or(8);
            if items.iter().any(|i| value_type_id(i) != ty) {
                bail!("GGUF arrays must be homogeneous");
            }
            w.write_all(&ty.to_le_bytes())?;
            w.write_all(&(items.len() as u64).to_le_bytes())?;
            for i in items {
                write_value(w, i)?;
            }
        }
    }
    Ok(())
}

fn pad_to<W: Write + Seek>(w: &mut W, align: u64) -> Result<()> {
    let pos = w.stream_position()?;
    let pad = (align - pos % align) % align;
    w.write_all(&vec![0u8; pad as usize])?;
    Ok(())
}

/// Write a GGUF file. `produce(i)` must return exactly `plans[i].nbytes()` bytes.
pub fn write_gguf(
    path: &Path,
    metadata: &[(String, Value)],
    plans: &[TensorPlan],
    mut produce: impl FnMut(usize) -> Result<Vec<u8>>,
) -> Result<()> {
    let align: u64 = metadata
        .iter()
        .find(|(k, _)| k == "general.alignment")
        .and_then(|(_, v)| v.to_u32().ok())
        .unwrap_or(32) as u64;
    let tmp = path.with_extension("gguf.part");
    let f = std::fs::File::create(&tmp)?;
    let mut w = std::io::BufWriter::with_capacity(1 << 22, f);
    w.write_all(b"GGUF")?;
    w.write_all(&3u32.to_le_bytes())?;
    w.write_all(&(plans.len() as u64).to_le_bytes())?;
    w.write_all(&(metadata.len() as u64).to_le_bytes())?;
    for (k, v) in metadata {
        write_str(&mut w, k)?;
        w.write_all(&value_type_id(v).to_le_bytes())?;
        write_value(&mut w, v)?;
    }
    let mut offset = 0u64;
    for p in plans {
        let elems: usize = p.dims.iter().product();
        if elems % p.dtype.block_size() != 0 {
            bail!(
                "tensor {} has {} elements, not a multiple of the {:?} block size",
                p.name,
                elems,
                p.dtype
            );
        }
        write_str(&mut w, &p.name)?;
        w.write_all(&(p.dims.len() as u32).to_le_bytes())?;
        for d in p.dims.iter().rev() {
            w.write_all(&(*d as u64).to_le_bytes())?;
        }
        w.write_all(&ggml_type_id(p.dtype).to_le_bytes())?;
        w.write_all(&offset.to_le_bytes())?;
        let n = p.nbytes() as u64;
        offset += n.div_ceil(align) * align;
    }
    pad_to(&mut w, align)?;
    for (i, p) in plans.iter().enumerate() {
        let data = produce(i)?;
        if data.len() != p.nbytes() {
            bail!(
                "tensor {}: produced {} bytes, expected {}",
                p.name,
                data.len(),
                p.nbytes()
            );
        }
        w.write_all(&data)?;
        pad_to(&mut w, align)?;
    }
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, path)?;
    Ok(())
}
