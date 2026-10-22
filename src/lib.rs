//! LLM fine-tuning from scratch in Rust.
//!
//! * [`model`]     — Llama-family transformer with frozen (optionally quantized) weights and LoRA.
//! * [`tokenizer`] — SentencePiece and byte-level BPE tokenizers read from GGUF / tokenizer.json.
//! * [`data`]      — PDF, Word, Excel, CSV, XML, JSON, HTML and text ingestion.
//! * [`train`]     — the training loop (AdamW, schedules, checkpointing, chunked loss).
//! * [`export`]    — PEFT / GGUF adapters and merged GGUF / safetensors models.
//! * [`server`]    — the web UI and its JSON API.

pub mod chat;
pub mod data;
pub mod device;
pub mod export;
pub mod generate;
pub mod gguf_writer;
pub mod model;
pub mod server;
pub mod tokenizer;
pub mod train;
pub mod util;
