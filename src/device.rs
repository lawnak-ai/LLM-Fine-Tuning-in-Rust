//! Device selection (CPU / CUDA / Metal).

use anyhow::{bail, Result};
use candle_core::Device;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceChoice {
    /// Use a GPU if one was compiled in and is present, otherwise the CPU.
    Auto,
    Cpu,
    Cuda(usize),
    Metal(usize),
}

impl DeviceChoice {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        let (kind, idx) = match s.split_once(':') {
            Some((k, i)) => (k.to_string(), i.parse::<usize>()?),
            None => (s.clone(), 0),
        };
        Ok(match kind.as_str() {
            "" | "auto" => Self::Auto,
            "cpu" => Self::Cpu,
            "cuda" | "gpu" | "nvidia" => Self::Cuda(idx),
            "metal" | "mps" | "apple" => Self::Metal(idx),
            other => bail!("unknown device '{other}', expected cpu, cuda[:N], metal or auto"),
        })
    }

    pub fn create(self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda(i) => {
                if !cfg!(feature = "cuda") {
                    bail!("this binary was built without CUDA support; rebuild with `cargo build --release --features cuda`")
                }
                let d = Device::new_cuda(i)?;
                check_cuda(&d)?;
                Ok(d)
            }
            Self::Metal(i) => {
                if !cfg!(feature = "metal") {
                    bail!("this binary was built without Metal support; rebuild with `cargo build --release --features metal`")
                }
                Ok(Device::new_metal(i)?)
            }
            Self::Auto => {
                if cuda_available() {
                    if let Ok(d) = Device::new_cuda(0) {
                        match check_cuda(&d) {
                            Ok(()) => return Ok(d),
                            Err(e) => eprintln!("{e}, using the CPU"),
                        }
                    }
                }
                if cfg!(feature = "metal") && candle_core::utils::metal_is_available() {
                    if let Ok(d) = Device::new_metal(0) {
                        return Ok(d);
                    }
                }
                Ok(Device::Cpu)
            }
        }
    }
}

/// Checks the GPU against the architecture the kernels were compiled for
/// (`CUDA_COMPUTE_CAP` at build time, e.g. 80 for the release build).
#[cfg(feature = "cuda")]
fn check_cuda(d: &Device) -> Result<()> {
    let Device::Cuda(cd) = d else { return Ok(()) };
    let Some(built) = option_env!("CUDA_COMPUTE_CAP")
        .and_then(|s| {
            s.trim_end_matches(|c: char| c.is_ascii_alphabetic())
                .parse::<i32>()
                .ok()
        })
        .map(|n| (n / 10, n % 10))
    else {
        // built for the local GPU
        return Ok(());
    };
    let cc = cd.cuda_stream().context().compute_capability()?;
    if cc < built {
        bail!(
            "this build needs an NVIDIA GPU with compute capability {}.{} or newer, this one has {}.{}",
            built.0,
            built.1,
            cc.0,
            cc.1
        );
    }
    // candle's fast GGUF matmul kernels only exist for the build architecture,
    // other GPUs use the dequantize + cuBLAS path
    if cc.0 != built.0 {
        candle_core::quantized::cuda::set_force_dmmv(true);
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn check_cuda(_: &Device) -> Result<()> {
    Ok(())
}

/// True when the NVIDIA driver loads and sees a GPU (candle's helper only
/// says whether CUDA was compiled in).
#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    use candle_core::cuda_backend::cudarc::driver::CudaContext;
    CudaContext::device_count().is_ok_and(|n| n > 0)
}

#[cfg(not(feature = "cuda"))]
fn cuda_available() -> bool {
    false
}

pub fn device_label(d: &Device) -> String {
    match d {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(_) => "cuda".to_string(),
        Device::Metal(_) => "metal".to_string(),
    }
}

#[derive(Serialize)]
pub struct SystemInfo {
    pub version: &'static str,
    pub cpu_threads: usize,
    pub avx: bool,
    pub f16c: bool,
    pub neon: bool,
    pub mkl: bool,
    pub accelerate: bool,
    pub cuda_compiled: bool,
    pub cuda_available: bool,
    pub metal_compiled: bool,
    pub metal_available: bool,
}

pub fn system_info() -> SystemInfo {
    SystemInfo {
        version: env!("CARGO_PKG_VERSION"),
        cpu_threads: candle_core::utils::get_num_threads(),
        avx: candle_core::utils::with_avx(),
        f16c: candle_core::utils::with_f16c(),
        neon: candle_core::utils::with_neon(),
        mkl: candle_core::utils::has_mkl(),
        accelerate: candle_core::utils::has_accelerate(),
        cuda_compiled: cfg!(feature = "cuda"),
        cuda_available: cuda_available(),
        metal_compiled: cfg!(feature = "metal"),
        metal_available: cfg!(feature = "metal") && candle_core::utils::metal_is_available(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_devices() {
        assert_eq!(DeviceChoice::parse("cpu").unwrap(), DeviceChoice::Cpu);
        assert_eq!(
            DeviceChoice::parse("CUDA:1").unwrap(),
            DeviceChoice::Cuda(1)
        );
        assert_eq!(DeviceChoice::parse("gpu").unwrap(), DeviceChoice::Cuda(0));
        assert_eq!(
            DeviceChoice::parse("metal").unwrap(),
            DeviceChoice::Metal(0)
        );
        assert_eq!(DeviceChoice::parse("").unwrap(), DeviceChoice::Auto);
        assert!(DeviceChoice::parse("tpu").is_err());
    }
}
