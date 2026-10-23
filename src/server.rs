//! HTTP server: the UI page and the JSON API.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::{Path as FsPath, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{anyhow, bail, Context};
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Multipart, Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, patch, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use tokio_stream::StreamExt;

use crate::chat::{ChatFormatter, Conversation, TemplateKind};
use crate::data::{ColumnMapping, Dataset};
use crate::device::{device_label, system_info, DeviceChoice};
use crate::export;
use crate::generate::{generate, GenParams};
use crate::model::load::{load_model, BasePrecision};
use crate::model::Model;
use crate::tokenizer::Tokenizer;
use crate::train::{self, TrainConfig, TrainJob, TrainStatus};
use crate::util::{human_bytes, human_count, path_within, sanitize_file_name};

const INDEX_HTML: &str = include_str!("ui/index.html");

#[derive(Default, Clone, serde::Serialize)]
pub struct TaskStatus {
    pub name: String,
    pub running: bool,
    pub message: String,
    pub error: Option<String>,
    pub result: Option<Value>,
}

#[derive(Default)]
struct Inner {
    model: Option<Arc<RwLock<Model>>>,
    tokenizer: Option<Arc<Tokenizer>>,
    model_info: Option<Value>,
    datasets: Vec<Dataset>,
    next_id: u64,
}

pub struct AppState {
    pub work_dir: PathBuf,
    pub models_dir: PathBuf,
    pub outputs_dir: PathBuf,
    inner: Mutex<Inner>,
    train_status: Arc<Mutex<TrainStatus>>,
    train_stop: Arc<AtomicBool>,
    gen_stop: Arc<AtomicBool>,
    generating: Arc<AtomicBool>,
    task: Arc<Mutex<TaskStatus>>,
}

type St = State<Arc<AppState>>;

pub struct ApiError(anyhow::Error);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": format!("{:#}", self.0)})),
        )
            .into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError(e.into())
    }
}

type ApiResult<T> = Result<T, ApiError>;

impl AppState {
    pub fn new(work_dir: PathBuf, models_dir: Option<PathBuf>) -> anyhow::Result<Arc<Self>> {
        let models_dir = models_dir.unwrap_or_else(|| work_dir.join("models"));
        let outputs_dir = work_dir.join("outputs");
        std::fs::create_dir_all(&models_dir)?;
        std::fs::create_dir_all(&outputs_dir)?;
        Ok(Arc::new(Self {
            work_dir,
            models_dir,
            outputs_dir,
            inner: Mutex::new(Inner::default()),
            train_status: Arc::new(Mutex::new(TrainStatus::idle())),
            train_stop: Arc::new(AtomicBool::new(false)),
            gen_stop: Arc::new(AtomicBool::new(false)),
            generating: Arc::new(AtomicBool::new(false)),
            task: Arc::new(Mutex::new(TaskStatus::default())),
        }))
    }

    fn model(&self) -> anyhow::Result<(Arc<RwLock<Model>>, Arc<Tokenizer>)> {
        let inner = self.inner.lock().unwrap();
        match (&inner.model, &inner.tokenizer) {
            (Some(m), Some(t)) => Ok((m.clone(), t.clone())),
            _ => bail!("no model loaded"),
        }
    }

    fn training_active(&self) -> bool {
        self.train_status.lock().unwrap().is_active()
    }

    /// Run a long blocking job in the background, reporting through `/api/task`.
    fn spawn_task<F>(self: &Arc<Self>, name: &str, f: F) -> anyhow::Result<()>
    where
        F: FnOnce(&dyn Fn(String)) -> anyhow::Result<Value> + Send + 'static,
    {
        {
            let mut t = self.task.lock().unwrap();
            if t.running {
                bail!("'{}' is still running", t.name);
            }
            *t = TaskStatus {
                name: name.to_string(),
                running: true,
                message: "starting".into(),
                error: None,
                result: None,
            };
        }
        let task = self.task.clone();
        std::thread::spawn(move || {
            let progress = |m: String| {
                eprintln!("[task] {m}");
                task.lock().unwrap().message = m;
            };
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&progress)));
            let mut t = task.lock().unwrap();
            t.running = false;
            match res {
                Ok(Ok(v)) => {
                    t.message = "done".into();
                    t.result = Some(v);
                }
                Ok(Err(e)) => {
                    t.message = "failed".into();
                    t.error = Some(format!("{e:#}"));
                }
                Err(_) => {
                    t.message = "failed".into();
                    t.error = Some("internal error, see the server log".into());
                }
            }
        });
        Ok(())
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/system", get(api_system))
        .route("/api/task", get(api_task))
        .route("/api/models", get(api_models))
        .route("/api/models/upload", post(api_model_upload))
        .route("/api/model", get(api_model_info))
        .route("/api/model/load", post(api_model_load))
        .route("/api/model/unload", post(api_model_unload))
        .route("/api/data", get(api_data_list))
        .route("/api/data/upload", post(api_data_upload))
        .route("/api/data/text", post(api_data_text))
        .route(
            "/api/data/{id}",
            patch(api_data_update).delete(api_data_delete),
        )
        .route("/api/data/{id}/sample", get(api_data_sample))
        .route("/api/train/start", post(api_train_start))
        .route("/api/train/stop", post(api_train_stop))
        .route("/api/train/status", get(api_train_status))
        .route("/api/train/defaults", get(api_train_defaults))
        .route("/api/generate", post(api_generate))
        .route("/api/generate/stop", post(api_generate_stop))
        .route("/api/adapter/save", post(api_adapter_save))
        .route("/api/adapter/load", post(api_adapter_load))
        .route("/api/adapter/remove", post(api_adapter_remove))
        .route("/api/export", post(api_export))
        .route("/api/files", get(api_files))
        .route("/api/download/{*path}", get(api_download))
        .layer(DefaultBodyLimit::disable())
        .with_state(state)
}

pub async fn serve(
    state: Arc<AppState>,
    addr: SocketAddr,
    open_browser: bool,
) -> anyhow::Result<()> {
    let app = router(state);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("cannot listen on {addr}"))?;
    let url = format!("http://{addr}");
    eprintln!("\n  LLM Fine-Tuning Studio is running:  {url}\n  Press Ctrl+C to stop.\n");
    if open_browser {
        crate::util::open_browser(&url);
    }
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn api_system(State(st): St) -> Json<Value> {
    let inner = st.inner.lock().unwrap();
    Json(json!({
        "system": system_info(),
        "work_dir": st.work_dir.display().to_string(),
        "models_dir": st.models_dir.display().to_string(),
        "outputs_dir": st.outputs_dir.display().to_string(),
        "model": inner.model_info,
        "datasets": inner.datasets.len(),
        "training": st.train_status.lock().unwrap().state,
    }))
}

async fn api_task(State(st): St) -> Json<TaskStatus> {
    Json(st.task.lock().unwrap().clone())
}

fn scan_models(dir: &FsPath, depth: usize, out: &mut Vec<Value>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if p.join("config.json").exists() && p.join("tokenizer.json").exists() {
                let size: u64 = std::fs::read_dir(&p)
                    .map(|r| {
                        r.filter_map(|e| e.ok())
                            .filter_map(|e| e.metadata().ok())
                            .map(|m| m.len())
                            .sum()
                    })
                    .unwrap_or(0);
                out.push(json!({"path": p.display().to_string(), "name": p.file_name().map(|n| n.to_string_lossy().to_string()), "kind": "safetensors", "size": human_bytes(size)}));
            } else if depth > 0 {
                scan_models(&p, depth - 1, out);
            }
        } else if p
            .extension()
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
        {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if name.contains("adapter") || name.ends_with(".part") {
                continue;
            }
            let size = p.metadata().map(|m| m.len()).unwrap_or(0);
            out.push(json!({"path": p.display().to_string(), "name": name, "kind": "gguf", "size": human_bytes(size)}));
        }
    }
}

async fn api_models(State(st): St) -> Json<Value> {
    let mut out = vec![];
    scan_models(&st.models_dir, 2, &mut out);
    let mut merged = vec![];
    scan_models(&st.outputs_dir, 2, &mut merged);
    out.extend(merged);
    Json(json!({"models": out, "models_dir": st.models_dir.display().to_string()}))
}

async fn api_model_upload(State(st): St, mut mp: Multipart) -> ApiResult<Json<Value>> {
    let mut saved = vec![];
    while let Some(mut field) = mp.next_field().await? {
        let Some(name) = field.file_name().map(sanitize_file_name) else {
            continue;
        };
        let lower = name.to_ascii_lowercase();
        let ok = [".gguf", ".safetensors", ".json", ".model", ".txt", ".jinja"]
            .iter()
            .any(|e| lower.ends_with(e));
        if !ok {
            return Err(anyhow!(
                "'{name}': upload a .gguf file (or the files of a Hugging Face model folder)"
            )
            .into());
        }
        let path = st.models_dir.join(&name);
        let tmp = st.models_dir.join(format!("{name}.part"));
        let mut f = tokio::fs::File::create(&tmp).await?;
        let mut n = 0u64;
        while let Some(chunk) = field.chunk().await? {
            n += chunk.len() as u64;
            f.write_all(&chunk).await?;
        }
        f.flush().await?;
        drop(f);
        tokio::fs::rename(&tmp, &path).await?;
        saved.push(json!({"path": path.display().to_string(), "size": human_bytes(n)}));
    }
    Ok(Json(json!({"saved": saved})))
}

fn model_info(model: &Model, tok: &Tokenizer, path: &FsPath, precision: &str) -> Value {
    let c = &model.cfg;
    let fmt = ChatFormatter::new(TemplateKind::Auto, tok);
    json!({
        "path": path.display().to_string(),
        "source": if model.is_gguf() { "gguf" } else { "safetensors" },
        "name": c.name,
        "arch": c.arch,
        "params": model.num_params(),
        "params_h": human_count(model.num_params()),
        "layers": c.num_layers,
        "hidden": c.hidden_size,
        "intermediate": c.intermediate_size,
        "heads": c.num_heads,
        "kv_heads": c.num_kv_heads,
        "head_dim": c.head_dim,
        "vocab": c.vocab_size,
        "context": c.context_length,
        "weights": model.weight_types(),
        "memory": human_bytes(model.weights_size_in_bytes() as u64),
        "device": device_label(&model.device),
        "precision": precision,
        "tokenizer": tok.info(),
        "chat_template": fmt.name(),
        "chat_template_error": fmt.jinja_error,
        "adapter": model.lora.as_ref().map(|l| json!({"rank": l.rank, "alpha": l.alpha, "targets": l.targets, "params": model.trainable_params()})),
    })
}

fn refresh_model_info(st: &AppState) {
    let mut inner = st.inner.lock().unwrap();
    let (Some(m), Some(t), Some(info)) = (
        inner.model.clone(),
        inner.tokenizer.clone(),
        inner.model_info.clone(),
    ) else {
        return;
    };
    let path = PathBuf::from(info["path"].as_str().unwrap_or(""));
    let prec = info["precision"].as_str().unwrap_or("auto").to_string();
    let v = match m.try_read() {
        Ok(m) => model_info(&m, &t, &path, &prec),
        Err(_) => return,
    };
    inner.model_info = Some(v);
}

async fn api_model_info(State(st): St) -> Json<Value> {
    refresh_model_info(&st);
    Json(json!({"model": st.inner.lock().unwrap().model_info}))
}

#[derive(Deserialize)]
struct LoadReq {
    path: String,
    #[serde(default)]
    device: String,
    #[serde(default)]
    precision: String,
}

async fn api_model_load(State(st): St, Json(req): Json<LoadReq>) -> ApiResult<Json<Value>> {
    if st.training_active() {
        return Err(anyhow!("cannot change the model while training is running").into());
    }
    let path = PathBuf::from(req.path.trim());
    let choice = DeviceChoice::parse(&req.device)?;
    let precision = BasePrecision::parse(&req.precision)?;
    let prec_name = if req.precision.is_empty() {
        "auto".to_string()
    } else {
        req.precision.clone()
    };
    let st2 = st.clone();
    st.spawn_task("load model", move |progress| {
        progress(format!("opening {}", path.display()));
        {
            // Free the previous model first.
            let mut inner = st2.inner.lock().unwrap();
            inner.model = None;
            inner.tokenizer = None;
            inner.model_info = None;
        }
        let device = choice.create()?;
        let t0 = std::time::Instant::now();
        let (model, tok) = load_model(&path, &device, precision, progress)?;
        let info = model_info(&model, &tok, &path, &prec_name);
        let mut inner = st2.inner.lock().unwrap();
        inner.model = Some(Arc::new(RwLock::new(model)));
        inner.tokenizer = Some(Arc::new(tok));
        inner.model_info = Some(info.clone());
        progress(format!("loaded in {:.1}s", t0.elapsed().as_secs_f64()));
        Ok(info)
    })?;
    Ok(Json(json!({"ok": true})))
}

async fn api_model_unload(State(st): St) -> ApiResult<Json<Value>> {
    if st.training_active() {
        return Err(anyhow!("training is running").into());
    }
    let mut inner = st.inner.lock().unwrap();
    inner.model = None;
    inner.tokenizer = None;
    inner.model_info = None;
    Ok(Json(json!({"ok": true})))
}

fn data_list(st: &AppState) -> Value {
    let inner = st.inner.lock().unwrap();
    let list: Vec<_> = inner.datasets.iter().map(|d| d.summary()).collect();
    json!({"datasets": list})
}

async fn api_data_list(State(st): St) -> Json<Value> {
    Json(data_list(&st))
}

async fn api_data_upload(State(st): St, mut mp: Multipart) -> ApiResult<Json<Value>> {
    let mut errors = vec![];
    let mut added = 0;
    while let Some(field) = mp.next_field().await? {
        let Some(name) = field.file_name().map(|s| s.to_string()) else {
            continue;
        };
        let bytes = field.bytes().await?;
        let id = {
            let mut inner = st.inner.lock().unwrap();
            inner.next_id += 1;
            format!("d{}", inner.next_id)
        };
        let name2 = name.clone();
        let res =
            tokio::task::spawn_blocking(move || Dataset::from_bytes(id, &name2, &bytes)).await?;
        match res {
            Ok(d) => {
                st.inner.lock().unwrap().datasets.push(d);
                added += 1;
            }
            Err(e) => errors.push(json!({"file": name, "error": format!("{e:#}")})),
        }
    }
    let mut v = data_list(&st);
    v["added"] = json!(added);
    v["errors"] = json!(errors);
    Ok(Json(v))
}

#[derive(Deserialize)]
struct TextReq {
    #[serde(default)]
    name: String,
    text: String,
}

async fn api_data_text(State(st): St, Json(req): Json<TextReq>) -> ApiResult<Json<Value>> {
    if req.text.trim().is_empty() {
        return Err(anyhow!("the text is empty").into());
    }
    {
        let mut inner = st.inner.lock().unwrap();
        inner.next_id += 1;
        let id = format!("d{}", inner.next_id);
        let name = if req.name.trim().is_empty() {
            "pasted text".to_string()
        } else {
            req.name.clone()
        };
        inner
            .datasets
            .push(Dataset::from_text(id, &name, &req.text));
    }
    Ok(Json(data_list(&st)))
}

#[derive(Deserialize)]
struct DataUpdate {
    enabled: Option<bool>,
    mapping: Option<ColumnMapping>,
}

async fn api_data_update(
    State(st): St,
    Path(id): Path<String>,
    Json(req): Json<DataUpdate>,
) -> ApiResult<Json<Value>> {
    {
        let mut inner = st.inner.lock().unwrap();
        let d = inner
            .datasets
            .iter_mut()
            .find(|d| d.id == id)
            .context("dataset not found")?;
        if let Some(e) = req.enabled {
            d.enabled = e;
        }
        if let Some(m) = req.mapping {
            d.mapping = m;
        }
    }
    Ok(Json(data_list(&st)))
}

async fn api_data_delete(State(st): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    st.inner.lock().unwrap().datasets.retain(|d| d.id != id);
    Ok(Json(data_list(&st)))
}

#[derive(Deserialize)]
struct SampleQuery {
    #[serde(default)]
    template: String,
    #[serde(default)]
    system: String,
}

/// Show how the first example of a dataset is presented to the model.
async fn api_data_sample(
    State(st): St,
    Path(id): Path<String>,
    Query(q): Query<SampleQuery>,
) -> ApiResult<Json<Value>> {
    let (_, tok) = st.model()?;
    let ds = {
        let inner = st.inner.lock().unwrap();
        inner
            .datasets
            .iter()
            .find(|d| d.id == id)
            .cloned()
            .context("dataset not found")?
    };
    let (texts, convs) = ds.examples();
    let fmt = ChatFormatter::new(TemplateKind::parse(&q.template)?, &tok);
    if let Some(mut c) = convs.into_iter().next() {
        if c.system.is_none() && !q.system.trim().is_empty() {
            c.system = Some(q.system.clone());
        }
        let r = fmt.render(&c, false)?;
        let (ids, mask) = crate::chat::tokenize_rendered(&tok, &fmt, &r, false);
        let trained = mask.iter().filter(|m| **m).count();
        let parts: Vec<Value> = segments(&r.text, &r.train_spans)
            .into_iter()
            .map(|(t, train)| json!({"text": t, "train": train}))
            .collect();
        return Ok(Json(
            json!({"template": fmt.name(), "parts": parts, "tokens": ids.len(), "trained_tokens": trained}),
        ));
    }
    let text = texts.into_iter().next().unwrap_or_default();
    let ids = tok.encode(&text, true, false);
    let shown: String = text.chars().take(3000).collect();
    Ok(Json(
        json!({"template": "none (plain text)", "parts": [{"text": shown, "train": true}], "tokens": ids.len(), "trained_tokens": ids.len()}),
    ))
}

fn segments(text: &str, spans: &[(usize, usize)]) -> Vec<(String, bool)> {
    let mut out = vec![];
    let mut pos = 0;
    for &(s, e) in spans {
        if s > pos {
            out.push((text[pos..s].to_string(), false));
        }
        out.push((text[s..e].to_string(), true));
        pos = e;
    }
    if pos < text.len() {
        out.push((text[pos..].to_string(), false));
    }
    out
}

async fn api_train_defaults() -> Json<TrainConfig> {
    Json(TrainConfig::default())
}

async fn api_train_start(State(st): St, Json(cfg): Json<TrainConfig>) -> ApiResult<Json<Value>> {
    if st.training_active() {
        return Err(anyhow!("training is already running").into());
    }
    if st.generating.load(Ordering::Relaxed) {
        return Err(anyhow!("wait for the playground generation to finish").into());
    }
    if st.task.lock().unwrap().running {
        return Err(anyhow!("wait for the current task to finish").into());
    }
    let (model, tok) = st.model()?;
    let datasets: Vec<Dataset> = st
        .inner
        .lock()
        .unwrap()
        .datasets
        .iter()
        .filter(|d| d.enabled)
        .cloned()
        .collect();
    if datasets.is_empty() {
        return Err(anyhow!("add some training data first").into());
    }
    let out = if cfg.output_dir.trim().is_empty() {
        st.outputs_dir
            .join(format!("run-{}", crate::util::timestamp_slug()))
    } else {
        let p = PathBuf::from(cfg.output_dir.trim());
        if p.is_absolute() {
            p
        } else {
            st.outputs_dir
                .join(sanitize_file_name(cfg.output_dir.trim()))
        }
    };
    *st.train_status.lock().unwrap() = TrainStatus {
        state: "preparing".into(),
        ..Default::default()
    };
    st.train_stop.store(false, Ordering::Relaxed);
    let job = TrainJob {
        model,
        tokenizer: tok,
        datasets,
        cfg,
        status: st.train_status.clone(),
        stop: st.train_stop.clone(),
        output_dir: out.clone(),
    };
    std::thread::Builder::new()
        .name("trainer".into())
        .spawn(move || train::run(job))?;
    Ok(Json(
        json!({"ok": true, "output_dir": out.display().to_string()}),
    ))
}

async fn api_train_stop(State(st): St) -> Json<Value> {
    st.train_stop.store(true, Ordering::Relaxed);
    let mut s = st.train_status.lock().unwrap();
    if s.state == "training" || s.state == "preparing" {
        s.state = "stopping".into();
        s.log("stop requested, finishing the current step");
    }
    Json(json!({"ok": true}))
}

#[derive(Deserialize)]
struct StatusQuery {
    #[serde(default)]
    since: usize,
    #[serde(default)]
    logs_since: usize,
}

async fn api_train_status(State(st): St, Query(q): Query<StatusQuery>) -> Json<Value> {
    let s = st.train_status.lock().unwrap().clone();
    let history: Vec<_> = s.history.iter().skip(q.since).cloned().collect();
    let logs: Vec<_> = s.logs.iter().skip(q.logs_since).cloned().collect();
    let mut v = serde_json::to_value(&s).unwrap_or(json!({}));
    v["history"] = json!(history);
    v["history_len"] = json!(s.history.len());
    v["logs"] = json!(logs);
    v["logs_len"] = json!(s.logs.len());
    Json(v)
}

#[derive(Deserialize)]
struct GenReq {
    prompt: String,
    #[serde(default)]
    system: String,
    #[serde(default)]
    history: Vec<(String, String)>,
    #[serde(default = "yes")]
    use_template: bool,
    #[serde(default)]
    template: String,
    #[serde(default)]
    params: GenParams,
}

fn yes() -> bool {
    true
}

async fn api_generate(State(st): St, Json(req): Json<GenReq>) -> ApiResult<Response> {
    if st.training_active() {
        return Err(anyhow!("the model is busy training").into());
    }
    if st.generating.swap(true, Ordering::SeqCst) {
        return Err(anyhow!("a generation is already running").into());
    }
    let (model, tok) = match st.model() {
        Ok(v) => v,
        Err(e) => {
            st.generating.store(false, Ordering::SeqCst);
            return Err(e.into());
        }
    };
    st.gen_stop.store(false, Ordering::SeqCst);
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(256);
    let stop = st.gen_stop.clone();
    let generating = st.generating.clone();
    tokio::task::spawn_blocking(move || {
        let send = |v: Value| tx.blocking_send(v.to_string() + "\n").is_ok();
        let res = (|| -> anyhow::Result<()> {
            let ids = if req.use_template {
                let fmt = ChatFormatter::new(TemplateKind::parse(&req.template)?, &tok);
                let mut conv = Conversation {
                    system: Some(req.system.clone()).filter(|s| !s.trim().is_empty()),
                    turns: vec![],
                };
                for (u, a) in &req.history {
                    conv.turns.push((u.clone(), Some(a.clone())));
                }
                conv.turns.push((req.prompt.clone(), None));
                let r = fmt.render(&conv, true)?;
                tok.encode(&r.text, !fmt.starts_with_bos(&r.text), true)
            } else {
                tok.encode(&req.prompt, true, true)
            };
            let model = model.read().map_err(|_| anyhow!("model lock poisoned"))?;
            let stats = generate(&model, &tok, &ids, &req.params, &stop, |t| {
                send(json!({"text": t}))
            })?;
            send(json!({"done": true, "stats": stats}));
            Ok(())
        })();
        if let Err(e) = res {
            send(json!({"error": format!("{e:#}")}));
        }
        generating.store(false, Ordering::SeqCst);
    });
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(Ok::<_, Infallible>);
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))?)
}

async fn api_generate_stop(State(st): St) -> Json<Value> {
    st.gen_stop.store(true, Ordering::SeqCst);
    Json(json!({"ok": true}))
}

#[derive(Deserialize, Default)]
struct PathReq {
    #[serde(default)]
    path: String,
}

async fn api_adapter_save(State(st): St, Json(req): Json<PathReq>) -> ApiResult<Json<Value>> {
    let (model, tok) = st.model()?;
    let dir = if req.path.trim().is_empty() {
        st.outputs_dir
            .join(format!("adapter-{}", crate::util::timestamp_slug()))
    } else {
        PathBuf::from(req.path.trim())
    };
    let m = model.try_read().map_err(|_| anyhow!("the model is busy"))?;
    let mut files = export::save_peft_adapter(&m, &tok, &dir)?;
    files.push(export::save_gguf_adapter(&m, &dir.join("adapter.gguf"))?);
    Ok(Json(
        json!({"dir": dir.display().to_string(), "files": files.iter().map(|f| f.display().to_string()).collect::<Vec<_>>()}),
    ))
}

async fn api_adapter_load(State(st): St, Json(req): Json<PathReq>) -> ApiResult<Json<Value>> {
    if st.training_active() {
        return Err(anyhow!("training is running").into());
    }
    let (model, _) = st.model()?;
    let msg = {
        let mut m = model
            .try_write()
            .map_err(|_| anyhow!("the model is busy"))?;
        export::load_adapter(&mut m, FsPath::new(req.path.trim()))?
    };
    refresh_model_info(&st);
    Ok(Json(json!({"message": msg})))
}

async fn api_adapter_remove(State(st): St) -> ApiResult<Json<Value>> {
    if st.training_active() {
        return Err(anyhow!("training is running").into());
    }
    let (model, _) = st.model()?;
    model
        .try_write()
        .map_err(|_| anyhow!("the model is busy"))?
        .detach_lora();
    refresh_model_info(&st);
    Ok(Json(json!({"ok": true})))
}

#[derive(Deserialize)]
struct ExportReq {
    /// merged_gguf | merged_safetensors
    kind: String,
    #[serde(default)]
    quant: String,
    #[serde(default)]
    name: String,
}

async fn api_export(State(st): St, Json(req): Json<ExportReq>) -> ApiResult<Json<Value>> {
    if st.training_active() {
        return Err(anyhow!("wait for training to finish").into());
    }
    let (model, tok) = st.model()?;
    if model.read().map(|m| m.lora.is_none()).unwrap_or(true) {
        return Err(anyhow!("no adapter to merge, train or load one first").into());
    }
    let quant = export::parse_export_quant(&req.quant)?;
    let base = if req.name.trim().is_empty() {
        format!("merged-{}", crate::util::timestamp_slug())
    } else {
        sanitize_file_name(req.name.trim())
    };
    let out_dir = st.outputs_dir.clone();
    let kind = req.kind.clone();
    st.spawn_task("export", move |progress| {
        let m = model.read().map_err(|_| anyhow!("model lock poisoned"))?;
        let path = match kind.as_str() {
            "merged_gguf" => {
                let name = if base.ends_with(".gguf") {
                    base.clone()
                } else {
                    format!("{base}.gguf")
                };
                export::export_merged_gguf(&m, &tok, &out_dir.join(name), quant, progress)?
            }
            "merged_safetensors" => {
                export::export_merged_safetensors(&m, &out_dir.join(&base), progress)?
            }
            other => bail!("unknown export kind '{other}'"),
        };
        let rel = path
            .strip_prefix(&out_dir)
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        Ok(json!({"path": path.display().to_string(), "download": rel}))
    })?;
    Ok(Json(json!({"ok": true})))
}

fn list_files(root: &FsPath, dir: &FsPath, out: &mut Vec<Value>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            list_files(root, &p, out);
        } else if let Ok(meta) = p.metadata() {
            let rel = p.strip_prefix(root).unwrap_or(&p).display().to_string();
            if rel.ends_with(".part") {
                continue;
            }
            out.push(json!({"path": rel, "size": human_bytes(meta.len()), "modified": meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs())}));
        }
    }
}

async fn api_files(State(st): St) -> Json<Value> {
    let mut out = vec![];
    list_files(&st.outputs_dir, &st.outputs_dir, &mut out);
    Json(json!({"files": out, "root": st.outputs_dir.display().to_string()}))
}

async fn api_download(State(st): St, Path(path): Path<String>) -> ApiResult<Response> {
    let full = path_within(&st.outputs_dir, &path).context("file not found")?;
    if !full.is_file() {
        return Err(anyhow!("file not found").into());
    }
    let file = tokio::fs::File::open(&full).await?;
    let len = file.metadata().await?.len();
    let name = full
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "download".into());
    let stream = tokio_util::io::ReaderStream::with_capacity(file, 1 << 20);
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_LENGTH, len)
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{}\"", name.replace('"', "")),
        )
        .body(Body::from_stream(stream))?)
}
