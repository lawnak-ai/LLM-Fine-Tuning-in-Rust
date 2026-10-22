//! Datasets, column mapping and building training samples.

pub mod extract;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::chat::{tokenize_rendered, ChatFormatter, Conversation};
use crate::tokenizer::Tokenizer;
use crate::util::Rng;
pub use extract::Extracted;

/// Which record columns become the prompt / answer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColumnMapping {
    pub prompt: Option<String>,
    /// Extra context appended to the prompt (Alpaca "input").
    pub input: Option<String>,
    pub response: Option<String>,
    pub system: Option<String>,
    /// Columns used as plain text when there is no prompt/response pair.
    pub text: Vec<String>,
}

const PROMPT_COLS: &[&str] = &[
    "instruction",
    "prompt",
    "question",
    "query",
    "user",
    "human",
    "request",
    "q",
    "input_text",
    "problem",
];
const INPUT_COLS: &[&str] = &["input", "context"];
const RESPONSE_COLS: &[&str] = &[
    "output",
    "response",
    "answer",
    "completion",
    "assistant",
    "gpt",
    "target",
    "reply",
    "a",
    "chosen",
    "solution",
];
const SYSTEM_COLS: &[&str] = &["system", "system_prompt"];
const TEXT_COLS: &[&str] = &["text", "content", "body", "document", "passage", "article"];

fn find_col(columns: &[String], names: &[&str], exclude: &[&Option<String>]) -> Option<String> {
    for n in names {
        for c in columns {
            if c.trim().eq_ignore_ascii_case(n)
                && !exclude.iter().any(|e| e.as_deref() == Some(c.as_str()))
            {
                return Some(c.clone());
            }
        }
    }
    None
}

pub fn auto_mapping(columns: &[String]) -> ColumnMapping {
    let mut m = ColumnMapping::default();
    m.prompt = find_col(columns, PROMPT_COLS, &[]);
    m.response = find_col(columns, RESPONSE_COLS, &[&m.prompt]);
    if m.prompt.is_none() && m.response.is_some() {
        m.prompt = find_col(columns, INPUT_COLS, &[&m.response]);
    } else if m.prompt.is_some() {
        m.input = find_col(columns, INPUT_COLS, &[&m.prompt, &m.response]);
    }
    m.system = find_col(columns, SYSTEM_COLS, &[&m.prompt, &m.response]);
    if m.prompt.is_none() || m.response.is_none() {
        m.prompt = None;
        m.response = None;
        m.input = None;
        m.text = match find_col(columns, TEXT_COLS, &[]) {
            Some(c) => vec![c],
            None => columns.to_vec(),
        };
    }
    m
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub size_bytes: usize,
    pub content: Extracted,
    pub mapping: ColumnMapping,
    pub enabled: bool,
}

#[derive(Serialize)]
pub struct DatasetSummary {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub size_bytes: usize,
    pub enabled: bool,
    /// "text", "records" or "conversations"
    pub content: &'static str,
    pub documents: usize,
    pub conversations: usize,
    pub characters: usize,
    pub columns: Vec<String>,
    pub mapping: ColumnMapping,
    pub preview: Vec<serde_json::Value>,
}

impl Dataset {
    pub fn from_bytes(id: String, name: &str, bytes: &[u8]) -> Result<Self> {
        let content = extract::extract(name, bytes)?;
        let mapping = match &content {
            Extracted::Records { columns, .. } => auto_mapping(columns),
            _ => ColumnMapping::default(),
        };
        Ok(Self {
            id,
            name: name.to_string(),
            kind: extract::file_kind(name),
            size_bytes: bytes.len(),
            content,
            mapping,
            enabled: true,
        })
    }

    pub fn from_text(id: String, name: &str, text: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            kind: "text".into(),
            size_bytes: text.len(),
            content: Extracted::Text(vec![extract::clean_text(text)]),
            mapping: ColumnMapping::default(),
            enabled: true,
        }
    }

    /// Plain text documents and conversations according to the mapping.
    pub fn examples(&self) -> (Vec<String>, Vec<Conversation>) {
        match &self.content {
            Extracted::Text(t) => (
                t.iter().filter(|s| !s.trim().is_empty()).cloned().collect(),
                vec![],
            ),
            Extracted::Conversations(c) => (vec![], c.clone()),
            Extracted::Records { columns, rows } => {
                let idx = |name: &Option<String>| {
                    name.as_ref()
                        .and_then(|n| columns.iter().position(|c| c == n))
                };
                let m = &self.mapping;
                if let (Some(p), Some(r)) = (idx(&m.prompt), idx(&m.response)) {
                    let inp = idx(&m.input);
                    let sys = idx(&m.system);
                    let mut convs = vec![];
                    for row in rows {
                        let mut prompt = row[p].trim().to_string();
                        if let Some(i) = inp {
                            let extra = row[i].trim();
                            if !extra.is_empty() {
                                prompt = format!("{prompt}\n\n{extra}");
                            }
                        }
                        let resp = row[r].trim().to_string();
                        if prompt.is_empty() || resp.is_empty() {
                            continue;
                        }
                        let system = sys
                            .map(|s| row[s].trim().to_string())
                            .filter(|s| !s.is_empty());
                        convs.push(Conversation::single(system, prompt, Some(resp)));
                    }
                    (vec![], convs)
                } else {
                    let cols: Vec<usize> = m
                        .text
                        .iter()
                        .filter_map(|t| columns.iter().position(|c| c == t))
                        .collect();
                    let cols = if cols.is_empty() {
                        (0..columns.len()).collect()
                    } else {
                        cols
                    };
                    let texts = rows
                        .iter()
                        .map(|row| {
                            if cols.len() == 1 {
                                row[cols[0]].trim().to_string()
                            } else {
                                cols.iter()
                                    .filter(|&&c| !row[c].trim().is_empty())
                                    .map(|&c| format!("{}: {}", columns[c], row[c].trim()))
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            }
                        })
                        .filter(|s| !s.is_empty())
                        .collect();
                    (texts, vec![])
                }
            }
        }
    }

    pub fn summary(&self) -> DatasetSummary {
        let (texts, convs) = self.examples();
        let characters = texts.iter().map(|t| t.chars().count()).sum::<usize>()
            + convs
                .iter()
                .map(|c| {
                    c.turns
                        .iter()
                        .map(|(u, a)| {
                            u.chars().count() + a.as_ref().map(|a| a.chars().count()).unwrap_or(0)
                        })
                        .sum::<usize>()
                })
                .sum::<usize>();
        let mut preview = vec![];
        for c in convs.iter().take(3) {
            let (u, a) = &c.turns[0];
            preview.push(serde_json::json!({"system": c.system, "prompt": truncate(u, 400), "response": truncate(a.as_deref().unwrap_or(""), 400)}));
        }
        for t in texts.iter().take(if convs.is_empty() { 3 } else { 0 }) {
            preview.push(serde_json::json!({"text": truncate(t, 600)}));
        }
        let (content, columns) = match &self.content {
            Extracted::Text(_) => ("text", vec![]),
            Extracted::Records { columns, .. } => ("records", columns.clone()),
            Extracted::Conversations(_) => ("conversations", vec![]),
        };
        DatasetSummary {
            id: self.id.clone(),
            name: self.name.clone(),
            kind: self.kind.clone(),
            size_bytes: self.size_bytes,
            enabled: self.enabled,
            content,
            documents: texts.len(),
            conversations: convs.len(),
            characters,
            columns,
            mapping: self.mapping.clone(),
            preview,
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(n).collect::<String>())
    }
}

/// One training sequence. `mask[i]` is true when token `i` is a prediction target.
#[derive(Debug, Clone)]
pub struct Sample {
    pub ids: Vec<u32>,
    pub mask: Vec<bool>,
}

impl Sample {
    pub fn targets(&self) -> usize {
        self.mask.iter().skip(1).filter(|m| **m).count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataMode {
    /// Documents -> language modelling, Q&A pairs -> chat fine-tuning.
    Auto,
    /// Everything as plain text (completion style).
    Text,
}

pub struct SampleOptions<'a> {
    pub max_len: usize,
    pub train_on_prompt: bool,
    pub system_prompt: Option<String>,
    pub mode: DataMode,
    pub formatter: &'a ChatFormatter,
}

#[derive(Debug, Default, Serialize, Clone)]
pub struct BuildStats {
    pub documents: usize,
    pub conversations: usize,
    pub text_chunks: usize,
    pub chat_samples: usize,
    pub truncated: usize,
    pub dropped: usize,
    pub total_tokens: usize,
    pub target_tokens: usize,
}

pub fn build_samples(
    datasets: &[Dataset],
    tok: &Tokenizer,
    opt: &SampleOptions,
) -> Result<(Vec<Sample>, BuildStats)> {
    if opt.max_len < 8 {
        bail!("max sequence length must be at least 8");
    }
    let mut stats = BuildStats::default();
    let mut texts = vec![];
    let mut convs = vec![];
    for d in datasets.iter().filter(|d| d.enabled) {
        let (t, c) = d.examples();
        texts.extend(t);
        convs.extend(c);
    }
    if opt.mode == DataMode::Text {
        for c in convs.drain(..) {
            let mut s = String::new();
            for (u, a) in &c.turns {
                s += u;
                s += "\n";
                if let Some(a) = a {
                    s += a;
                    s += "\n\n";
                }
            }
            texts.push(s);
        }
    }
    stats.documents = texts.len();
    stats.conversations = convs.len();
    let mut samples = vec![];

    // Documents: concatenate (BOS doc EOS BOS doc EOS ...) and cut into windows.
    let mut stream: Vec<u32> = vec![];
    for t in &texts {
        stream.extend(tok.encode(t, true, false));
        if let Some(e) = tok.eos {
            stream.push(e);
        }
    }
    let mut start = 0;
    while start < stream.len() {
        let end = (start + opt.max_len).min(stream.len());
        let ids = stream[start..end].to_vec();
        if ids.len() >= 8 || samples.is_empty() {
            let mut mask = vec![true; ids.len()];
            mask[0] = false;
            samples.push(Sample { ids, mask });
            stats.text_chunks += 1;
        }
        start = end;
    }

    // Conversations: chat template, loss only on answers.
    for mut c in convs {
        if c.system.is_none() {
            c.system = opt.system_prompt.clone().filter(|s| !s.trim().is_empty());
        }
        let r = opt.formatter.render(&c, false)?;
        let (mut ids, mut mask) = tokenize_rendered(tok, opt.formatter, &r, opt.train_on_prompt);
        if ids.len() > opt.max_len {
            truncate_sample(&mut ids, &mut mask, opt.max_len);
            stats.truncated += 1;
        }
        if mask.iter().skip(1).any(|m| *m) {
            samples.push(Sample { ids, mask });
            stats.chat_samples += 1;
        } else {
            stats.dropped += 1;
        }
    }
    if samples.is_empty() {
        bail!("no training data: upload files or paste some text first");
    }
    stats.total_tokens = samples.iter().map(|s| s.ids.len()).sum();
    stats.target_tokens = samples.iter().map(|s| s.targets()).sum();
    Ok((samples, stats))
}

/// Shorten a sample that is too long. Normally cut the end, but if that loses most of
/// the answer, drop prompt tokens (after BOS) instead.
fn truncate_sample(ids: &mut Vec<u32>, mask: &mut Vec<bool>, max_len: usize) {
    let trained = mask.iter().filter(|m| **m).count();
    let kept_by_cut = mask[..max_len].iter().filter(|m| **m).count();
    let first_train = mask.iter().position(|m| *m).unwrap_or(0);
    let overflow = ids.len() - max_len;
    if kept_by_cut * 4 < trained && first_train > overflow {
        ids.drain(1..1 + overflow);
        mask.drain(1..1 + overflow);
    } else {
        ids.truncate(max_len);
        mask.truncate(max_len);
    }
}

/// Split off an evaluation set.
pub fn split_eval(
    mut samples: Vec<Sample>,
    fraction: f64,
    seed: u64,
) -> (Vec<Sample>, Vec<Sample>) {
    if fraction <= 0.0 || samples.len() < 4 {
        return (samples, vec![]);
    }
    let mut rng = Rng::new(seed ^ 0xE7A1);
    rng.shuffle(&mut samples);
    let n_eval = ((samples.len() as f64 * fraction).round() as usize).clamp(1, samples.len() / 2);
    let eval = samples.split_off(samples.len() - n_eval);
    (samples, eval)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_detects_alpaca_and_qa() {
        let cols = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let m = auto_mapping(&cols(&["instruction", "input", "output"]));
        assert_eq!(m.prompt.as_deref(), Some("instruction"));
        assert_eq!(m.input.as_deref(), Some("input"));
        assert_eq!(m.response.as_deref(), Some("output"));
        let m = auto_mapping(&cols(&["Question", "Answer"]));
        assert_eq!(m.prompt.as_deref(), Some("Question"));
        assert_eq!(m.response.as_deref(), Some("Answer"));
        let m = auto_mapping(&cols(&["input", "target"]));
        assert_eq!(m.prompt.as_deref(), Some("input"));
        assert_eq!(m.response.as_deref(), Some("target"));
        let m = auto_mapping(&cols(&["name", "city"]));
        assert!(m.prompt.is_none());
        assert_eq!(m.text.len(), 2);
    }

    #[test]
    fn truncation_keeps_the_answer() {
        // BOS, 6 prompt tokens, 3 answer tokens; limit 6 -> prompt is shortened after BOS.
        let mut ids: Vec<u32> = (0..10).collect();
        let mut mask = vec![
            false, false, false, false, false, false, false, true, true, true,
        ];
        truncate_sample(&mut ids, &mut mask, 6);
        assert_eq!(ids, vec![0, 5, 6, 7, 8, 9]);
        assert_eq!(mask.iter().filter(|m| **m).count(), 3);
        // Long answer: plain cut at the end.
        let mut ids: Vec<u32> = (0..10).collect();
        let mut mask = vec![false, false, true, true, true, true, true, true, true, true];
        truncate_sample(&mut ids, &mut mask, 6);
        assert_eq!(ids, vec![0, 1, 2, 3, 4, 5]);
    }
}
