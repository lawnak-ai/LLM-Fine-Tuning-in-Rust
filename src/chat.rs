//! Chat templates: turn conversations into text, and mark which characters are
//! the assistant's answers (the only part the model is trained to produce).

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemplateKind {
    /// The model's own Jinja template if it renders cleanly, otherwise a detected built-in.
    Auto,
    /// Always the model's own Jinja template.
    Model,
    ChatMl,
    Llama3,
    Llama2,
    Mistral,
    Zephyr,
    Phi3,
    Alpaca,
    Plain,
}

impl TemplateKind {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "" | "auto" => Self::Auto,
            "model" | "jinja" => Self::Model,
            "chatml" | "qwen" => Self::ChatMl,
            "llama3" | "llama_3" => Self::Llama3,
            "llama2" | "llama_2" => Self::Llama2,
            "mistral" => Self::Mistral,
            "zephyr" | "tinyllama" => Self::Zephyr,
            "phi3" | "phi_3" => Self::Phi3,
            "alpaca" => Self::Alpaca,
            "plain" | "none" | "raw" => Self::Plain,
            other => return Err(anyhow!("unknown chat template '{other}'")),
        })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Conversation {
    pub system: Option<String>,
    /// (user, assistant). The last assistant may be `None` for a generation prompt.
    pub turns: Vec<(String, Option<String>)>,
}

impl Conversation {
    pub fn single(system: Option<String>, user: String, assistant: Option<String>) -> Self {
        Self { system, turns: vec![(user, assistant)] }
    }
}

/// Rendered text and the byte ranges that should be trained on.
pub struct Rendered {
    pub text: String,
    pub train_spans: Vec<(usize, usize)>,
}

pub struct ChatFormatter {
    kind: TemplateKind,
    jinja: Option<String>,
    bos_text: String,
    eos_text: String,
    /// Why the model's own template could not be used (if it could not).
    pub jinja_error: Option<String>,
}

fn detect_builtin(tpl: Option<&str>, tok: &Tokenizer) -> TemplateKind {
    if let Some(t) = tpl {
        if t.contains("<|im_start|>") {
            return TemplateKind::ChatMl;
        }
        if t.contains("<|start_header_id|>") {
            return TemplateKind::Llama3;
        }
        if t.contains("[INST]") {
            return if t.contains("<<SYS>>") { TemplateKind::Llama2 } else { TemplateKind::Mistral };
        }
        if t.contains("<|user|>") {
            return if t.contains("<|end|>") { TemplateKind::Phi3 } else { TemplateKind::Zephyr };
        }
    }
    if tok.token_id("<|im_start|>").is_some() {
        TemplateKind::ChatMl
    } else if tok.token_id("<|start_header_id|>").is_some() {
        TemplateKind::Llama3
    } else {
        TemplateKind::Alpaca
    }
}

const ALPACA_SYSTEM: &str =
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.";

impl ChatFormatter {
    pub fn new(kind: TemplateKind, tok: &Tokenizer) -> Self {
        let bos_text = tok.bos.map(|b| tok.token_text(b).to_string()).unwrap_or_default();
        let eos_text = tok.eos.map(|b| tok.token_text(b).to_string()).unwrap_or_default();
        let jinja = tok.chat_template.clone();
        let mut f = Self { kind, jinja: None, bos_text, eos_text, jinja_error: None };
        match kind {
            TemplateKind::Auto | TemplateKind::Model => {
                if let Some(t) = jinja {
                    f.jinja = Some(t);
                    f.kind = TemplateKind::Model;
                    match f.probe() {
                        Ok(()) => return f,
                        Err(e) => f.jinja_error = Some(format!("{e:#}")),
                    }
                    f.jinja = None;
                }
                f.kind = detect_builtin(tok.chat_template.as_deref(), tok);
            }
            _ => {}
        }
        f
    }

    /// Check that the Jinja template renders and that prompts are prefixes of the full text.
    fn probe(&self) -> Result<()> {
        let conv = Conversation {
            system: Some("You are helpful.".into()),
            turns: vec![("Hi".into(), Some("Hello!".into())), ("How are you?".into(), Some("Fine.".into()))],
        };
        let r = self.render(&conv, false)?;
        if r.train_spans.len() != 2 {
            return Err(anyhow!("template probe failed"));
        }
        let p = self.render(&Conversation::single(None, "Hi".into(), None), true)?;
        if !p.text.contains("Hi") {
            return Err(anyhow!("template probe failed"));
        }
        Ok(())
    }

    pub fn name(&self) -> String {
        format!("{:?}", self.kind).to_lowercase()
    }

    pub fn kind(&self) -> TemplateKind {
        self.kind
    }

    /// Whether the rendered text already starts with BOS.
    pub fn starts_with_bos(&self, text: &str) -> bool {
        !self.bos_text.is_empty() && text.starts_with(&self.bos_text)
    }

    pub fn render(&self, conv: &Conversation, add_generation_prompt: bool) -> Result<Rendered> {
        if self.kind == TemplateKind::Model {
            if let Some(t) = &self.jinja {
                return self.render_jinja(t, conv, add_generation_prompt);
            }
        }
        Ok(self.render_builtin(conv))
    }

    fn render_builtin(&self, conv: &Conversation) -> Rendered {
        let mut s = String::new();
        let mut spans = vec![];
        let eos = &self.eos_text;
        let sys = conv.system.as_deref().filter(|s| !s.trim().is_empty());
        for (i, (user, asst)) in conv.turns.iter().enumerate() {
            match self.kind {
                TemplateKind::ChatMl => {
                    if i == 0 {
                        if let Some(sys) = sys {
                            s += &format!("<|im_start|>system\n{sys}<|im_end|>\n");
                        }
                    }
                    s += &format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n");
                    if let Some(a) = asst {
                        let st = s.len();
                        s += a;
                        s += "<|im_end|>";
                        spans.push((st, s.len()));
                        s += "\n";
                    }
                }
                TemplateKind::Llama3 => {
                    if i == 0 {
                        if let Some(sys) = sys {
                            s += &format!("<|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>");
                        }
                    }
                    s += &format!(
                        "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    );
                    if let Some(a) = asst {
                        let st = s.len();
                        s += a;
                        s += "<|eot_id|>";
                        spans.push((st, s.len()));
                    }
                }
                TemplateKind::Llama2 => {
                    if i > 0 {
                        s += &self.bos_text;
                    }
                    match (i, sys) {
                        (0, Some(sys)) => s += &format!("[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user} [/INST]"),
                        _ => s += &format!("[INST] {user} [/INST]"),
                    }
                    if let Some(a) = asst {
                        let st = s.len();
                        s += &format!(" {a} {eos}");
                        spans.push((st, s.len()));
                    }
                }
                TemplateKind::Mistral => {
                    match (i, sys) {
                        (0, Some(sys)) => s += &format!("[INST] {sys}\n\n{user} [/INST]"),
                        _ => s += &format!("[INST] {user} [/INST]"),
                    }
                    if let Some(a) = asst {
                        let st = s.len();
                        s += a;
                        s += eos;
                        spans.push((st, s.len()));
                    }
                }
                TemplateKind::Zephyr => {
                    if i == 0 {
                        if let Some(sys) = sys {
                            s += &format!("<|system|>\n{sys}{eos}\n");
                        }
                    }
                    s += &format!("<|user|>\n{user}{eos}\n<|assistant|>\n");
                    if let Some(a) = asst {
                        let st = s.len();
                        s += a;
                        s += eos;
                        spans.push((st, s.len()));
                        s += "\n";
                    }
                }
                TemplateKind::Phi3 => {
                    if i == 0 {
                        if let Some(sys) = sys {
                            s += &format!("<|system|>\n{sys}<|end|>\n");
                        }
                    }
                    s += &format!("<|user|>\n{user}<|end|>\n<|assistant|>\n");
                    if let Some(a) = asst {
                        let st = s.len();
                        s += a;
                        s += "<|end|>";
                        spans.push((st, s.len()));
                        s += "\n";
                    }
                }
                TemplateKind::Alpaca => {
                    if i == 0 {
                        s += sys.unwrap_or(ALPACA_SYSTEM);
                        s += "\n\n";
                    }
                    s += &format!("### Instruction:\n{user}\n\n### Response:\n");
                    if let Some(a) = asst {
                        let st = s.len();
                        s += a;
                        s += eos;
                        spans.push((st, s.len()));
                        s += "\n\n";
                    }
                }
                TemplateKind::Plain | TemplateKind::Auto | TemplateKind::Model => {
                    if i == 0 {
                        if let Some(sys) = sys {
                            s += sys;
                            s += "\n\n";
                        }
                    }
                    s += user;
                    s += "\n";
                    if let Some(a) = asst {
                        let st = s.len();
                        s += a;
                        s += eos;
                        spans.push((st, s.len()));
                        s += "\n";
                    }
                }
            }
        }
        Rendered { text: s, train_spans: spans }
    }

    fn jinja_text(&self, tpl: &str, messages: &[serde_json::Value], add_generation_prompt: bool) -> Result<String> {
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |_fmt: String| -> String { "01 Jan 2026".to_string() });
        let t = env.template_from_str(tpl).map_err(|e| anyhow!("chat template error: {e}"))?;
        let ctx = minijinja::context! {
            messages => minijinja::Value::from(minijinja::value::Serde(messages)),
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos_text.clone(),
            eos_token => self.eos_text.clone(),
        };
        t.render(ctx).map_err(|e| anyhow!("chat template error: {e}"))
    }

    fn render_jinja(&self, tpl: &str, conv: &Conversation, add_generation_prompt: bool) -> Result<Rendered> {
        let mut msgs = vec![];
        if let Some(sys) = conv.system.as_deref().filter(|s| !s.trim().is_empty()) {
            msgs.push(serde_json::json!({"role": "system", "content": sys}));
        }
        let mut spans = vec![];
        for (user, asst) in &conv.turns {
            msgs.push(serde_json::json!({"role": "user", "content": user}));
            if let Some(a) = asst {
                let prefix = self.jinja_text(tpl, &msgs, true)?;
                msgs.push(serde_json::json!({"role": "assistant", "content": a}));
                let full = self.jinja_text(tpl, &msgs, false)?;
                if !full.starts_with(&prefix) {
                    return Err(anyhow!("chat template does not render prompts as prefixes"));
                }
                // The answer ends at the end-of-turn marker: drop trailing whitespace after it.
                let end = full.trim_end().len();
                spans.push((prefix.len(), end.max(prefix.len())));
            }
        }
        let gen = add_generation_prompt && conv.turns.last().map(|t| t.1.is_none()).unwrap_or(false);
        let text = self.jinja_text(tpl, &msgs, gen)?;
        for (s, e) in &spans {
            if *e > text.len() || *s > *e {
                return Err(anyhow!("chat template rendering is not incremental"));
            }
        }
        Ok(Rendered { text, train_spans: spans })
    }
}

/// Tokenize a rendered conversation and produce per-token training labels.
/// Returns `(ids, mask)` where `mask[i]` is true when token `i` should be predicted.
pub fn tokenize_rendered(tok: &Tokenizer, fmt: &ChatFormatter, r: &Rendered, train_on_prompt: bool) -> (Vec<u32>, Vec<bool>) {
    let add_special = !fmt.starts_with_bos(&r.text);
    let ids = tok.encode(&r.text, add_special, true);
    if train_on_prompt {
        let mut mask = vec![true; ids.len()];
        if let Some(m) = mask.first_mut() {
            *m = false;
        }
        return (ids, mask);
    }
    let mut mask = vec![false; ids.len()];
    for &(s, e) in &r.train_spans {
        let start = common_prefix(&tok.encode(&r.text[..s], add_special, true), &ids);
        let end = common_prefix(&tok.encode(&r.text[..e], add_special, true), &ids);
        let end = if end < start { start } else { end };
        // Tokens fully inside the prefix of the answer stay masked; everything up to `end` is trained.
        let end = if end == start && e > s { ids.len().min(start + 1) } else { end };
        for m in mask.iter_mut().take(end).skip(start) {
            *m = true;
        }
    }
    (ids, mask)
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}
