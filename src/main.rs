use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};

use llm_finetune::chat::{ChatFormatter, Conversation, TemplateKind};
use llm_finetune::data::Dataset;
use llm_finetune::device::{device_label, DeviceChoice};
use llm_finetune::export;
use llm_finetune::generate::{generate, GenParams};
use llm_finetune::model::load::{load_model, BasePrecision};
use llm_finetune::server::{self, AppState};
use llm_finetune::train::{self, TrainConfig, TrainJob, TrainStatus};
use llm_finetune::util::human_count;

#[derive(Parser)]
#[command(
    name = "llm-finetune",
    version,
    about = "LoRA fine-tuning for LLMs (web UI and CLI)"
)]
struct Cli {
    /// Number of CPU threads (default: all cores).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the web UI (default).
    Serve(ServeArgs),
    /// Fine-tune from the command line.
    Train(Box<TrainArgs>),
    /// Generate text with a model (and optional adapter).
    Generate(GenArgs),
    /// Merge an adapter into a model and write a GGUF / safetensors model.
    Merge(MergeArgs),
    /// Show model information.
    Info(ModelArgs),
    /// Tokenize text (debugging).
    Tokenize(TokenizeArgs),
}

#[derive(Args, Clone)]
struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 7860)]
    port: u16,
    /// Folder for models, uploads and outputs.
    #[arg(long, default_value = "workspace")]
    work_dir: PathBuf,
    /// Folder scanned for models (default: <work-dir>/models).
    #[arg(long)]
    models_dir: Option<PathBuf>,
    /// Don't open the browser on start.
    #[arg(long)]
    no_browser: bool,
}

impl Default for ServeArgs {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 7860,
            work_dir: "workspace".into(),
            models_dir: None,
            no_browser: false,
        }
    }
}

#[derive(Args)]
struct ModelArgs {
    /// .gguf file or Hugging Face model folder.
    #[arg(long, short)]
    model: PathBuf,
    /// cpu, cuda, cuda:1, metal or auto.
    #[arg(long, default_value = "auto")]
    device: String,
    /// Base weight storage: auto (keep), f32, f16, bf16, q8_0, q4_k, ...
    #[arg(long, default_value = "auto")]
    precision: String,
}

#[derive(Args)]
struct TrainArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Training files (pdf, docx, xlsx, csv, tsv, xml, json, jsonl, txt, md, html).
    #[arg(long, short, num_args = 1.., required = true)]
    data: Vec<PathBuf>,
    /// JSON file with a full training config (CLI flags override it).
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    rank: Option<usize>,
    #[arg(long)]
    alpha: Option<f64>,
    #[arg(long)]
    dropout: Option<f32>,
    /// Comma separated: q,k,v,o,gate,up,down (or all, attn, mlp).
    #[arg(long)]
    targets: Option<String>,
    #[arg(long)]
    lr: Option<f64>,
    #[arg(long)]
    epochs: Option<usize>,
    #[arg(long)]
    max_steps: Option<usize>,
    #[arg(long)]
    batch_size: Option<usize>,
    #[arg(long)]
    grad_accum: Option<usize>,
    #[arg(long)]
    seq_len: Option<usize>,
    /// auto, model, chatml, llama3, llama2, mistral, zephyr, phi3, alpaca, plain
    #[arg(long)]
    template: Option<String>,
    #[arg(long)]
    system: Option<String>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    eval_fraction: Option<f64>,
    /// Treat every file as plain text (no chat formatting).
    #[arg(long)]
    text_only: bool,
    /// Disable gradient checkpointing (faster, more memory).
    #[arg(long)]
    no_checkpointing: bool,
    #[arg(long, short, default_value = "workspace/outputs/cli-run")]
    output: PathBuf,
    /// Also write a merged GGUF with this quantization (e.g. q8_0, q4_k, f16, same).
    #[arg(long)]
    merge_gguf: Option<String>,
}

#[derive(Args)]
struct GenArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long)]
    adapter: Option<PathBuf>,
    #[arg(long, short)]
    prompt: String,
    #[arg(long)]
    system: Option<String>,
    /// Feed the prompt as-is instead of through the chat template.
    #[arg(long)]
    raw: bool,
    #[arg(long, default_value = "auto")]
    template: String,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[derive(Args)]
struct MergeArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long)]
    adapter: PathBuf,
    /// Output .gguf file, or a folder for safetensors.
    #[arg(long, short)]
    output: PathBuf,
    /// GGUF quantization: same, f16, q8_0, q4_k, q5_k, q6_k, ...
    #[arg(long, default_value = "same")]
    quant: String,
}

#[derive(Args)]
struct TokenizeArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long, short)]
    text: String,
    /// Do not add BOS / parse special tokens.
    #[arg(long)]
    no_special: bool,
}

fn load(
    args: &ModelArgs,
) -> Result<(
    llm_finetune::model::Model,
    llm_finetune::tokenizer::Tokenizer,
)> {
    let device = DeviceChoice::parse(&args.device)?.create()?;
    eprintln!("device: {}", device_label(&device));
    let (m, t) = load_model(
        &args.model,
        &device,
        BasePrecision::parse(&args.precision)?,
        &|s| eprintln!("{s}"),
    )?;
    eprintln!(
        "model: {} ({}, {} params, {} layers, weights {})",
        m.cfg.name,
        m.cfg.arch,
        human_count(m.num_params()),
        m.cfg.num_layers,
        llm_finetune::util::human_bytes(m.weights_size_in_bytes() as u64)
    );
    Ok((m, t))
}

fn main() {
    // started by double-click: keep the window open so errors can be read
    let no_args = std::env::args_os().len() == 1;
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        if no_args {
            eprintln!("\nPress Enter to close.");
            let _ = std::io::stdin().read_line(&mut String::new());
        }
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    if let Some(n) = cli.threads {
        std::env::set_var("RAYON_NUM_THREADS", n.to_string());
    }
    match cli.cmd.unwrap_or(Cmd::Serve(ServeArgs::default())) {
        Cmd::Serve(a) => {
            let addr: SocketAddr = format!("{}:{}", a.host, a.port)
                .parse()
                .context("bad host/port")?;
            let state = AppState::new(a.work_dir, a.models_dir)?;
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(server::serve(state, addr, !a.no_browser))
        }
        Cmd::Info(a) => {
            let (m, t) = load(&a)?;
            println!("{}", serde_json::to_string_pretty(&m.cfg)?);
            println!("{}", serde_json::to_string_pretty(&t.info())?);
            let fmt = ChatFormatter::new(TemplateKind::Auto, &t);
            println!("chat template: {}", fmt.name());
            if let Some(e) = &fmt.jinja_error {
                println!("(the model's own template could not be used: {e})");
            }
            println!("weight types: {:?}", m.weight_types());
            Ok(())
        }
        Cmd::Tokenize(a) => {
            let (_, t) = load(&a.model)?;
            let ids = t.encode(&a.text, !a.no_special, !a.no_special);
            println!("{ids:?}");
            for id in &ids {
                println!("{id:>8}  {:?}", t.token_text(*id));
            }
            Ok(())
        }
        Cmd::Generate(a) => {
            let (mut m, t) = load(&a.model)?;
            if let Some(ad) = &a.adapter {
                eprintln!("{}", export::load_adapter(&mut m, ad)?);
            }
            let ids = if a.raw {
                t.encode(&a.prompt, true, true)
            } else {
                let fmt = ChatFormatter::new(TemplateKind::parse(&a.template)?, &t);
                let r = fmt.render(
                    &Conversation::single(a.system.clone(), a.prompt.clone(), None),
                    true,
                )?;
                t.encode(&r.text, !fmt.starts_with_bos(&r.text), true)
            };
            let p = GenParams {
                max_new_tokens: a.max_tokens,
                temperature: a.temperature,
                top_p: a.top_p,
                top_k: a.top_k,
                repeat_penalty: a.repeat_penalty,
                seed: a.seed,
                ..Default::default()
            };
            use std::io::Write;
            let stats = generate(&m, &t, &ids, &p, &AtomicBool::new(false), |s| {
                print!("{s}");
                std::io::stdout().flush().ok();
                true
            })?;
            println!();
            eprintln!(
                "[{} prompt tokens in {:.2}s, {} tokens at {:.1} tok/s, stop: {}]",
                stats.prompt_tokens,
                stats.prompt_secs,
                stats.generated_tokens,
                stats.tokens_per_sec,
                stats.stop_reason
            );
            Ok(())
        }
        Cmd::Merge(a) => {
            let (mut m, t) = load(&a.model)?;
            eprintln!("{}", export::load_adapter(&mut m, &a.adapter)?);
            let is_gguf = a.output.extension().map(|e| e == "gguf").unwrap_or(false);
            let out = if is_gguf {
                export::export_merged_gguf(
                    &m,
                    &t,
                    &a.output,
                    export::parse_export_quant(&a.quant)?,
                    &|s| eprintln!("{s}"),
                )?
            } else {
                export::export_merged_safetensors(&m, &a.output, &|s| eprintln!("{s}"))?
            };
            eprintln!("wrote {}", out.display());
            Ok(())
        }
        Cmd::Train(a) => {
            let mut cfg: TrainConfig = match &a.config {
                Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
                None => TrainConfig::default(),
            };
            macro_rules! set {
                ($field:ident, $v:expr) => {
                    if let Some(v) = $v {
                        cfg.$field = v;
                    }
                };
            }
            set!(lora_rank, a.rank);
            set!(lora_alpha, a.alpha);
            set!(lora_dropout, a.dropout);
            set!(learning_rate, a.lr);
            set!(epochs, a.epochs);
            set!(max_steps, a.max_steps);
            set!(batch_size, a.batch_size);
            set!(grad_accum, a.grad_accum);
            set!(max_seq_len, a.seq_len);
            set!(chat_template, a.template.clone());
            set!(system_prompt, a.system.clone());
            set!(seed, a.seed);
            set!(eval_fraction, a.eval_fraction);
            if let Some(t) = &a.targets {
                cfg.target_modules = t
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            if a.text_only {
                cfg.data_mode = "text".into();
            }
            if a.no_checkpointing {
                cfg.gradient_checkpointing = false;
            }
            let mut datasets = vec![];
            for (i, p) in a.data.iter().enumerate() {
                let bytes =
                    std::fs::read(p).with_context(|| format!("cannot read {}", p.display()))?;
                let name = p
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                let d = Dataset::from_bytes(format!("d{i}"), &name, &bytes)?;
                let s = d.summary();
                eprintln!(
                    "data: {name}: {} documents, {} conversations, {} characters",
                    s.documents, s.conversations, s.characters
                );
                datasets.push(d);
            }
            let (m, t) = load(&a.model)?;
            let model = Arc::new(RwLock::new(m));
            let tok = Arc::new(t);
            let status = Arc::new(Mutex::new(TrainStatus::idle()));
            let job = TrainJob {
                model: model.clone(),
                tokenizer: tok.clone(),
                datasets,
                cfg,
                status: status.clone(),
                stop: Arc::new(AtomicBool::new(false)),
                output_dir: a.output.clone(),
            };
            train::run(job);
            let st = status.lock().unwrap().clone();
            if st.state == "failed" {
                bail!("training failed: {}", st.message);
            }
            if let Some(q) = &a.merge_gguf {
                let m = model.read().unwrap();
                let out = a.output.join("merged.gguf");
                export::export_merged_gguf(&m, &tok, &out, export::parse_export_quant(q)?, &|s| {
                    eprintln!("{s}")
                })?;
                eprintln!("merged model: {}", out.display());
            }
            Ok(())
        }
    }
}
