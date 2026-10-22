//! Chat templates: turn conversations into text, and mark which characters are
//! the assistant's answers (the only part the model is trained to produce).

use anyhow::{anyhow, Context, Result};
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
    /// Texts of the end-of-generation tokens (`<|im_end|>`, `<|eot_id|>`, `</s>`, ...).
    eog_texts: Vec<String>,
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

/// Remove Hugging Face's `{% generation %}` markers, which only exist for assistant masks.
fn clean_template(t: &str) -> String {
    match fancy_regex::Regex::new(r"\{%-?\s*(?:end)?generation\s*-?%\}") {
        Ok(re) => re.replace_all(t, "").into_owned(),
        Err(_) => t.to_string(),
    }
}

const ALPACA_SYSTEM: &str =
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.";

impl ChatFormatter {
    pub fn new(kind: TemplateKind, tok: &Tokenizer) -> Self {
        let bos_text = tok.bos.map(|b| tok.token_text(b).to_string()).unwrap_or_default();
        let eos_text = tok.eos.map(|b| tok.token_text(b).to_string()).unwrap_or_default();
        let jinja = tok.chat_template.clone();
        let eog_texts = tok.end_of_generation_ids().iter().map(|&i| tok.token_text(i).to_string()).filter(|t| !t.is_empty()).collect();
        let mut f = Self { kind, jinja: None, bos_text, eos_text, jinja_error: None, eog_texts };
        match kind {
            TemplateKind::Auto | TemplateKind::Model => {
                if let Some(t) = jinja {
                    f.jinja = Some(clean_template(&t));
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

    /// Length of the end-of-turn marker that follows an answer in `rest`.
    fn end_marker_len(&self, suffix: &str, rest: &str) -> usize {
        let mut cands: Vec<(usize, &str)> = self
            .eog_texts
            .iter()
            .filter_map(|t| suffix.find(t.as_str()).map(|p| (p, &suffix[..p + t.len()])))
            .collect();
        cands.sort();
        let first_line = suffix.split('\n').next().unwrap_or("").trim_end();
        let extra = [first_line, suffix];
        for c in cands.iter().map(|c| c.1).chain(extra) {
            if !c.is_empty() && rest.starts_with(c) {
                return c.len();
            }
        }
        // Earlier turns may end with a different token than the last one (e.g. <|end|> vs <|return|>).
        self.eog_texts.iter().filter(|t| rest.starts_with(t.as_str())).map(|t| t.len()).max().unwrap_or(0)
    }

    fn render_jinja(&self, tpl: &str, conv: &Conversation, add_generation_prompt: bool) -> Result<Rendered> {
        let has_system = conv.system.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false);
        match self.render_jinja_with(tpl, conv, add_generation_prompt, false) {
            Ok(r) => Ok(r),
            // Some templates (Gemma, ...) reject a system role: fold it into the first user turn.
            Err(e) if has_system => self.render_jinja_with(tpl, conv, add_generation_prompt, true).map_err(|_| e),
            Err(e) => Err(e),
        }
    }

    fn render_jinja_with(&self, tpl: &str, conv: &Conversation, add_generation_prompt: bool, merge_system: bool) -> Result<Rendered> {
        let sys = conv.system.as_deref().filter(|s| !s.trim().is_empty());
        let mut msgs = vec![];
        if let (Some(s), false) = (sys, merge_system) {
            msgs.push(serde_json::json!({"role": "system", "content": s}));
        }
        for (i, (user, asst)) in conv.turns.iter().enumerate() {
            let user = match (sys, merge_system && i == 0) {
                (Some(s), true) => format!("{s}\n\n{user}"),
                _ => user.clone(),
            };
            msgs.push(serde_json::json!({"role": "user", "content": user}));
            if let Some(a) = asst {
                msgs.push(serde_json::json!({"role": "assistant", "content": a}));
            }
        }
        let gen = add_generation_prompt && conv.turns.last().map(|t| t.1.is_none()).unwrap_or(false);
        let text = self.jinja_text(tpl, &msgs, gen)?;

        // Find every answer in the final text (templates may render earlier turns differently
        // from the last one, so offsets from partial renders cannot be reused).
        let mut spans = vec![];
        let mut from = 0;
        for (i, m) in msgs.iter().enumerate() {
            let content = m["content"].as_str().unwrap_or("").trim();
            if content.is_empty() {
                continue;
            }
            match m["role"].as_str() {
                Some("user") => {
                    if let Some(p) = text[from..].find(content) {
                        from += p + content.len();
                    }
                }
                Some("assistant") => {
                    let start = from + text[from..].find(content).context("the chat template changes the answer text")?;
                    let mut end = start + content.len();
                    // The end-of-turn marker is whatever follows the answer when it is the last
                    // message, cut after the first end-of-generation token (templates may append
                    // more, e.g. tool instructions).
                    let partial = self.jinja_text(tpl, &msgs[..=i], false)?;
                    let suffix = partial.rfind(content).map(|q| partial[q + content.len()..].trim_end().to_string()).unwrap_or_default();
                    end += self.end_marker_len(&suffix, &text[end..]);
                    spans.push((start, end));
                    from = end;
                }
                _ => {}
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(kind: TemplateKind, jinja: Option<&str>, eog: &[&str]) -> ChatFormatter {
        ChatFormatter {
            kind,
            jinja: jinja.map(clean_template),
            bos_text: String::new(),
            eos_text: "</s>".into(),
            jinja_error: None,
            eog_texts: eog.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn trained(r: &Rendered) -> Vec<&str> {
        r.train_spans.iter().map(|&(s, e)| &r.text[s..e]).collect()
    }

    fn two_turns() -> Conversation {
        Conversation {
            system: Some("Be brief.".into()),
            turns: vec![("Hi".into(), Some("Hello!".into())), ("2+2?".into(), Some("4".into()))],
        }
    }

    #[test]
    fn builtin_chatml_spans() {
        let r = fmt(TemplateKind::ChatMl, None, &[]).render(&two_turns(), false).unwrap();
        assert_eq!(trained(&r), vec!["Hello!<|im_end|>", "4<|im_end|>"]);
        assert!(r.text.starts_with("<|im_start|>system\nBe brief.<|im_end|>\n"));
        let p = fmt(TemplateKind::ChatMl, None, &[]).render(&Conversation::single(None, "Q".into(), None), true).unwrap();
        assert!(p.text.ends_with("<|im_start|>assistant\n"));
        assert!(p.train_spans.is_empty());
    }

    #[test]
    fn jinja_spans_follow_the_final_text() {
        // Like Qwen3: only the last assistant message gets an empty think block.
        let tpl = "{% for m in messages %}<|im_start|>{{ m.role }}\n{% if m.role == 'assistant' and loop.last %}<think>\n\n</think>\n\n{% endif %}{% generation %}{{ m.content }}{% endgeneration %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        let f = fmt(TemplateKind::Model, Some(tpl), &["<|im_end|>"]);
        let r = f.render(&two_turns(), false).unwrap();
        assert_eq!(trained(&r), vec!["Hello!<|im_end|>", "4<|im_end|>"]);
    }

    #[test]
    fn system_role_is_merged_when_rejected() {
        let tpl = "{% for m in messages %}{% if m.role == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}<start_of_turn>{{ m.role }}\n{{ m.content }}<end_of_turn>\n{% endfor %}";
        let f = fmt(TemplateKind::Model, Some(tpl), &["<end_of_turn>"]);
        let r = f.render(&two_turns(), false).unwrap();
        assert!(r.text.contains("Be brief.\n\nHi"));
        assert_eq!(trained(&r), vec!["Hello!<end_of_turn>", "4<end_of_turn>"]);
    }

    #[test]
    fn end_marker_stops_at_end_of_generation_token() {
        let f = fmt(TemplateKind::Model, None, &["<|END_OF_TURN_TOKEN|>", "<|end|>", "<|return|>"]);
        let tool = "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write 'Action:'";
        assert_eq!(f.end_marker_len(tool, tool), "<|END_OF_TURN_TOKEN|>".len());
        // Earlier turns end with <|end|> although the last one ends with <|return|>.
        assert_eq!(f.end_marker_len("<|return|>", "<|end|><|start|>user"), "<|end|>".len());
        let g = fmt(TemplateKind::Model, None, &[]);
        assert_eq!(g.end_marker_len("\n</assistant>", "\n</assistant>\n<user>"), "\n</assistant>".len());
        assert_eq!(g.end_marker_len("<|return|>", "<|end|>"), 0);
    }
}
