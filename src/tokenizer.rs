//! A from-scratch tokenizer that understands the two vocabularies used by
//! llama-family models:
//!
//! * SentencePiece style (`tokenizer.ggml.model = "llama"`): score driven
//!   bigram merging with byte fallback (Llama 1/2, Mistral, TinyLlama, ...).
//! * Byte-level BPE (`tokenizer.ggml.model = "gpt2"`): regex pre-tokenisation,
//!   GPT-2 byte to unicode mapping and rank ordered merges (Llama 3, Qwen,
//!   SmolLM, ...).
//!
//! Vocabularies are read either from GGUF metadata or from a Hugging Face
//! `tokenizer.json`. The algorithms follow llama.cpp so that token ids match
//! what llama.cpp / Ollama / LM Studio produce for the same model.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Value;
use fancy_regex::Regex;
use serde::Serialize;
use serde_json::Value as Json;

pub const TOKEN_NORMAL: i32 = 1;
pub const TOKEN_UNKNOWN: i32 = 2;
pub const TOKEN_CONTROL: i32 = 3;
pub const TOKEN_USER_DEFINED: i32 = 4;
pub const TOKEN_UNUSED: i32 = 5;
pub const TOKEN_BYTE: i32 = 6;

const SPACE_ESCAPE: char = '\u{2581}'; // ▁

// Pre-tokenizer regexes (same as llama.cpp, see src/llama-vocab.cpp).
const RE_GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";
const RE_GPT2_FULL: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const RE_LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_QWEN35: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_GPT4O: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_TEKKEN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_SEED_CODER: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+";
const RE_DEEPSEEK3: &str = r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##;
const RE_BLOOM: &str = r" ?[^(\s|.,!?…。，、।۔،)]+";

/// Regexes, `ignore_merges`, and an `add_bos` override for a GGUF `tokenizer.ggml.pre` name.
fn pre_tokenizer_for(name: &str) -> (Vec<&'static str>, bool, Option<bool>) {
    match name {
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "midm-2.0" | "lfm2"
        | "jina-v5-nano" => (vec![RE_LLAMA3], true, Some(true)),
        "dbrx" | "smaug-bpe" | "chatglm-bpe" | "glm4" | "glm5" => (vec![RE_LLAMA3], false, None),
        "qwen2" | "deepseek-r1-qwen" | "stablelm2" | "hunyuan" | "megrez" | "kormo" | "f2llmv2"
        | "grok-2" | "solar-open" => (vec![RE_QWEN2], false, None),
        "qwen35" => (vec![RE_QWEN35], false, None),
        "gpt-4o" | "llama4" | "kanana2" | "minimax-m2" | "talkie" => (vec![RE_GPT4O], false, None),
        "tekken" | "pixtral" => (vec![RE_TEKKEN], false, None),
        "seed-coder" => (vec![RE_SEED_CODER], false, None),
        "starcoder" | "refact" | "command-r" | "smollm" | "codeshell" | "exaone" | "minerva-7b" => {
            (vec![r"\p{N}", RE_GPT2], false, None)
        }
        "falcon" => (vec![r"[\p{P}\$\+<=>\^~\|`]+", RE_GPT2, r"[0-9][0-9][0-9]"], false, None),
        "deepseek-coder" => (
            vec![r"[\r\n]", r"\s?\p{L}+", r"\s?\p{P}+", r"[一-龥ࠀ-一가-퟿]+", r"\p{N}"],
            false,
            None,
        ),
        "deepseek-llm" => (
            vec![
                r"[\r\n]",
                r"\s?[A-Za-z\x{B5}\x{C0}-\x{D6}\x{D8}-\x{F6}\x{F8}-\x{1BA}\x{1BC}-\x{1BF}\x{1C4}-\x{293}\x{295}-\x{2AF}\x{370}-\x{373}\x{376}\x{377}\x{37B}-\x{37D}\x{37F}\x{386}\x{388}-\x{38A}\x{38C}\x{38E}-\x{3A1}\x{3A3}-\x{3F5}\x{3F7}-\x{481}\x{48A}-\x{52F}\x{531}-\x{556}\x{10A0}-\x{10C5}\x{13A0}-\x{13F5}\x{13F8}-\x{13FD}\x{1C90}-\x{1CBA}\x{1CBD}-\x{1CBF}\x{1D00}-\x{1D2B}\x{1D6B}-\x{1D77}\x{1D79}-\x{1D9A}\x{1E00}-\x{1F15}\x{1F18}-\x{1F1D}\x{1F20}-\x{1F45}\x{1F48}-\x{1F4D}\x{1F50}-\x{1F57}\x{1F59}\x{1F5B}\x{1F5D}\x{1F5F}-\x{1F7D}\x{1F80}-\x{1FB4}\x{1FB6}-\x{1FBC}\x{1FBE}\x{1FC2}-\x{1FC4}\x{1FC6}-\x{1FCC}\x{1FD0}-\x{1FD3}\x{1FD6}-\x{1FDB}\x{1FE0}-\x{1FEC}\x{1FF2}-\x{1FF4}\x{1FF6}-\x{1FFC}\x{2102}\x{2107}\x{210A}-\x{2113}\x{2115}\x{2119}-\x{211D}\x{2124}\x{2126}\x{2128}\x{212A}-\x{212D}\x{212F}-\x{2134}\x{2139}\x{213C}-\x{213F}\x{2145}-\x{2149}\x{214E}\x{2183}\x{2184}\x{2C00}-\x{2C7B}\x{2C7E}-\x{2CE4}\x{2CEB}-\x{2CEE}\x{2CF2}\x{2CF3}\x{A640}-\x{A66D}\x{A680}-\x{A69B}\x{A722}-\x{A76F}\x{A771}-\x{A787}\x{A78B}-\x{A78E}\x{AB70}-\x{ABBF}\x{FB00}-\x{FB06}\x{FB13}-\x{FB17}\x{FF21}-\x{FF3A}\x{FF41}-\x{FF5A}\x{10400}-\x{1044F}\x{104B0}-\x{104D3}\x{104D8}-\x{104FB}\x{10C80}-\x{10CB2}\x{10CC0}-\x{10CF2}\x{118A0}-\x{118DF}\x{1E900}-\x{1E943}]+",
                r"\s?[!-/:-~\x{FF01}-\x{FF0F}\x{FF1A}-\x{FF5E}\x{2018}-\x{201F}\x{3000}-\x{3002}]+",
                r"\s+$",
                r"[\x{4E00}-\x{9FA5}\x{800}-\x{4E00}\x{AC00}-\x{D7FF}]+",
                r"\p{N}+",
            ],
            false,
            None,
        ),
        "deepseek-v3" | "hunyuan-dense" | "joyai-llm" | "hy_v4" => {
            (vec![r"\p{N}{1,3}", r"[一-龥぀-ゟ゠-ヿ]+", RE_DEEPSEEK3], false, None)
        }
        "poro-chat" | "bloom" | "gpt3-finnish" => (vec![RE_BLOOM], false, None),
        "viking" => (vec![RE_BLOOM, r"\p{N}"], false, None),
        // "default", "gpt-2", "mpt", "olmo", "jais", "phi-2", ... and anything unknown.
        _ => (vec![RE_GPT2], false, None),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    /// SentencePiece vocabulary, merges driven by token scores.
    SpmScores,
    /// SentencePiece-like vocabulary (▁ for spaces, byte fallback) with BPE merge ranks.
    SpmMerges,
    /// GPT-2 style byte-level BPE.
    ByteBpe,
}

enum Fragment<'a> {
    Text(&'a str),
    Token(u32),
}

pub struct Tokenizer {
    kind: TokenizerKind,
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    scores: Vec<f32>,
    token_types: Vec<i32>,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    merges_text: Vec<String>,
    regexes: Vec<Regex>,
    pre_name: String,
    /// Special strings that are matched before normal tokenisation: (text, id, always matched).
    specials: Vec<(String, u32, bool)>,
    specials_by_first: HashMap<char, Vec<usize>>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
    pub unk: Option<u32>,
    pub eot: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
    add_space_prefix: bool,
    ignore_merges: bool,
    nfc: bool,
    byte_tokens: Vec<Option<u32>>,
    byte_encoder: Vec<char>,
    byte_decoder: HashMap<char, u8>,
    pub chat_template: Option<String>,
    cache: Mutex<HashMap<String, Vec<u32>>>,
}

fn bytes_to_unicode() -> Vec<char> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).collect();
    bs.extend(0xA1u32..=0xAC);
    bs.extend(0xAEu32..=0xFF);
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = vec!['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        table[*b as usize] = char::from_u32(*c).unwrap();
    }
    table
}

fn value_as_u32(v: &Value) -> Option<u32> {
    match v {
        Value::U8(x) => Some(*x as u32),
        Value::I8(x) => Some(*x as u32),
        Value::U16(x) => Some(*x as u32),
        Value::I16(x) => Some(*x as u32),
        Value::U32(x) => Some(*x),
        Value::I32(x) => Some(*x as u32),
        Value::U64(x) => Some(*x as u32),
        Value::I64(x) => Some(*x as u32),
        _ => None,
    }
}

fn value_as_i32(v: &Value) -> Option<i32> {
    value_as_u32(v).map(|x| x as i32)
}

fn value_as_f32(v: &Value) -> Option<f32> {
    match v {
        Value::F32(x) => Some(*x),
        Value::F64(x) => Some(*x as f32),
        other => value_as_u32(other).map(|x| x as f32),
    }
}

/// Information about the tokenizer for display in the UI.
#[derive(Serialize, Clone)]
pub struct TokenizerInfo {
    pub kind: TokenizerKind,
    pub pre: String,
    pub vocab_size: usize,
    pub merges: usize,
    pub bos: Option<String>,
    pub eos: Option<String>,
    pub add_bos: bool,
    pub has_chat_template: bool,
}

#[derive(PartialEq)]
struct SpmBigram {
    score: f32,
    left: usize,
    right: usize,
    size: usize,
}
impl Eq for SpmBigram {}
impl Ord for SpmBigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: higher score first, then the leftmost pair.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}
impl PartialOrd for SpmBigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(PartialEq, Eq)]
struct BpeBigram {
    rank: u32,
    left: usize,
    right: usize,
    left_id: u32,
    right_id: u32,
}
impl Ord for BpeBigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: lowest rank first, then the leftmost pair.
        other.rank.cmp(&self.rank).then_with(|| other.left.cmp(&self.left))
    }
}
impl PartialOrd for BpeBigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy)]
struct Sym {
    start: usize,
    end: usize,
    prev: Option<usize>,
    next: Option<usize>,
    id: Option<u32>,
}

fn init_symbols(text: &str) -> Vec<Sym> {
    let mut syms = Vec::with_capacity(text.len());
    let mut iter = text.char_indices().peekable();
    while let Some((start, ch)) = iter.next() {
        let end = start + ch.len_utf8();
        let idx = syms.len();
        syms.push(Sym {
            start,
            end,
            prev: if idx == 0 { None } else { Some(idx - 1) },
            next: if iter.peek().is_some() { Some(idx + 1) } else { None },
            id: None,
        });
    }
    syms
}

impl Tokenizer {
    #[allow(clippy::too_many_arguments)]
    fn build(
        kind: TokenizerKind,
        tokens: Vec<String>,
        scores: Vec<f32>,
        mut token_types: Vec<i32>,
        merges_text: Vec<String>,
        regexes: Vec<String>,
        pre_name: String,
    ) -> Result<Self> {
        let n = tokens.len();
        if token_types.len() != n {
            token_types = vec![TOKEN_NORMAL; n];
        }
        let mut token_to_id = HashMap::with_capacity(n);
        for (i, t) in tokens.iter().enumerate() {
            // Keep the first id when a string appears twice.
            token_to_id.entry(t.clone()).or_insert(i as u32);
        }
        let mut merges = HashMap::with_capacity(merges_text.len());
        for (rank, m) in merges_text.iter().enumerate() {
            let Some((l, r)) = m.split_once(' ') else { continue };
            let (Some(&li), Some(&ri)) = (token_to_id.get(l), token_to_id.get(r)) else { continue };
            let merged = format!("{l}{r}");
            let Some(&mi) = token_to_id.get(&merged) else { continue };
            merges.entry((li, ri)).or_insert((rank as u32, mi));
        }
        let mut compiled = Vec::new();
        for r in &regexes {
            compiled.push(Regex::new(r).with_context(|| format!("bad pre-tokenizer regex {r}"))?);
        }
        let byte_encoder = bytes_to_unicode();
        let byte_decoder = byte_encoder.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        let mut byte_tokens = vec![None; 256];
        for (b, slot) in byte_tokens.iter_mut().enumerate() {
            *slot = token_to_id.get(&format!("<0x{b:02X}>")).copied();
        }
        let mut tok = Self {
            kind,
            scores: if scores.len() == n { scores } else { vec![0.0; n] },
            tokens,
            token_to_id,
            token_types,
            merges,
            merges_text,
            regexes: compiled,
            pre_name,
            specials: vec![],
            specials_by_first: HashMap::new(),
            bos: None,
            eos: None,
            pad: None,
            unk: None,
            eot: None,
            add_bos: false,
            add_eos: false,
            add_space_prefix: false,
            ignore_merges: false,
            nfc: false,
            byte_tokens,
            byte_encoder,
            byte_decoder,
            chat_template: None,
            cache: Mutex::new(HashMap::new()),
        };
        tok.rebuild_specials();
        Ok(tok)
    }

    fn rebuild_specials(&mut self) {
        let mut specials: Vec<(String, u32, bool)> = self
            .tokens
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                let tt = self.token_types[i];
                if t.is_empty() {
                    return None;
                }
                match tt {
                    TOKEN_CONTROL | TOKEN_UNKNOWN => Some((t.clone(), i as u32, false)),
                    TOKEN_USER_DEFINED => Some((t.clone(), i as u32, true)),
                    _ => None,
                }
            })
            .collect();
        // Longest match wins.
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let mut by_first: HashMap<char, Vec<usize>> = HashMap::new();
        for (i, (s, _, _)) in specials.iter().enumerate() {
            if let Some(c) = s.chars().next() {
                by_first.entry(c).or_default().push(i);
            }
        }
        self.specials = specials;
        self.specials_by_first = by_first;
    }

    /// Build a tokenizer from GGUF metadata.
    pub fn from_gguf(md: &HashMap<String, Value>) -> Result<Self> {
        let model = md
            .get("tokenizer.ggml.model")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "llama".to_string());
        let tokens: Vec<String> = md
            .get("tokenizer.ggml.tokens")
            .context("GGUF file has no tokenizer (tokenizer.ggml.tokens missing)")?
            .to_vec()?
            .iter()
            .map(|v| v.to_string().cloned().unwrap_or_default())
            .collect();
        let scores: Vec<f32> = md
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.to_vec().ok())
            .map(|v| v.iter().map(|x| value_as_f32(x).unwrap_or(0.0)).collect())
            .unwrap_or_default();
        let token_types: Vec<i32> = md
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.to_vec().ok())
            .map(|v| v.iter().map(|x| value_as_i32(x).unwrap_or(TOKEN_NORMAL)).collect())
            .unwrap_or_default();
        let merges: Vec<String> = md
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.to_vec().ok())
            .map(|v| v.iter().map(|x| x.to_string().cloned().unwrap_or_default()).collect())
            .unwrap_or_default();
        let pre = md
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "default".to_string());
        let get_id = |k: &str| md.get(k).and_then(value_as_u32);
        let get_bool = |k: &str| md.get(k).and_then(|v| v.to_bool().ok());

        let mut tok = match model.as_str() {
            "llama" => {
                let mut t = Self::build(
                    TokenizerKind::SpmScores,
                    tokens,
                    scores,
                    token_types,
                    vec![],
                    vec![],
                    "spm".to_string(),
                )?;
                t.add_bos = true;
                t.add_space_prefix = true;
                t
            }
            "gpt2" => {
                let (res, ignore_merges, add_bos) = pre_tokenizer_for(&pre);
                let mut t = Self::build(
                    TokenizerKind::ByteBpe,
                    tokens,
                    scores,
                    token_types,
                    merges,
                    res.iter().map(|s| s.to_string()).collect(),
                    pre.clone(),
                )?;
                t.ignore_merges = ignore_merges;
                t.add_bos = add_bos.unwrap_or(false);
                t
            }
            other => bail!(
                "unsupported tokenizer type '{other}' (supported: llama/SentencePiece and gpt2/BPE)"
            ),
        };
        tok.bos = get_id("tokenizer.ggml.bos_token_id");
        tok.eos = get_id("tokenizer.ggml.eos_token_id");
        tok.unk = get_id("tokenizer.ggml.unknown_token_id");
        tok.pad = get_id("tokenizer.ggml.padding_token_id");
        tok.eot = get_id("tokenizer.ggml.eot_token_id");
        if let Some(b) = get_bool("tokenizer.ggml.add_bos_token") {
            tok.add_bos = b;
        }
        if let Some(b) = get_bool("tokenizer.ggml.add_eos_token") {
            tok.add_eos = b;
        }
        if let Some(b) = get_bool("tokenizer.ggml.add_space_prefix") {
            tok.add_space_prefix = b;
        }
        if tok.add_bos && tok.bos.is_none() {
            tok.add_bos = false;
        }
        tok.chat_template = md
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string().ok().cloned());
        Ok(tok)
    }

    /// Build a tokenizer from Hugging Face `tokenizer.json` (+ optional `tokenizer_config.json`).
    pub fn from_hf_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        let j: Json = serde_json::from_str(&raw).context("invalid tokenizer.json")?;
        let cfg: Json = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Json::Null);
        let model_cfg: Json = std::fs::read_to_string(dir.join("config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Json::Null);
        let mut tok = Self::from_hf_json(&j, &cfg, &model_cfg)?;
        if tok.chat_template.is_none() {
            if let Ok(t) = std::fs::read_to_string(dir.join("chat_template.jinja")) {
                tok.chat_template = Some(t);
            } else if let Some(t) = std::fs::read_to_string(dir.join("chat_template.json"))
                .ok()
                .and_then(|s| serde_json::from_str::<Json>(&s).ok())
                .and_then(|j| j.get("chat_template").and_then(|t| t.as_str()).map(String::from))
            {
                tok.chat_template = Some(t);
            }
        }
        Ok(tok)
    }

    pub fn from_hf_json(j: &Json, cfg: &Json, model_cfg: &Json) -> Result<Self> {
        let model = &j["model"];
        let mtype = model["type"].as_str().unwrap_or("BPE");
        if mtype != "BPE" {
            bail!("unsupported tokenizer.json model type '{mtype}' (only BPE is supported)");
        }
        let vocab = model["vocab"].as_object().context("tokenizer.json: model.vocab missing")?;
        let mut max_id = vocab.values().filter_map(|v| v.as_u64()).max().unwrap_or(0) as usize;
        let added = j["added_tokens"].as_array().cloned().unwrap_or_default();
        for a in &added {
            if let Some(id) = a["id"].as_u64() {
                max_id = max_id.max(id as usize);
            }
        }
        let mut tokens = vec![String::new(); max_id + 1];
        let mut types = vec![TOKEN_UNUSED; max_id + 1];
        for (s, id) in vocab {
            if let Some(id) = id.as_u64() {
                tokens[id as usize] = s.clone();
                types[id as usize] = TOKEN_NORMAL;
            }
        }
        for a in &added {
            let (Some(id), Some(content)) = (a["id"].as_u64(), a["content"].as_str()) else {
                continue;
            };
            tokens[id as usize] = content.to_string();
            types[id as usize] = if a["special"].as_bool().unwrap_or(false) {
                TOKEN_CONTROL
            } else {
                TOKEN_USER_DEFINED
            };
        }
        let merges: Vec<String> = model["merges"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| match m {
                        Json::String(s) => Some(s.clone()),
                        Json::Array(p) if p.len() == 2 => {
                            Some(format!("{} {}", p[0].as_str()?, p[1].as_str()?))
                        }
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Walk normalizer / pre-tokenizer / decoder to find out what flavour this is.
        let mut regexes: Vec<String> = vec![];
        let mut byte_level = false;
        let mut metaspace = false;
        let mut prefix_space = false;
        let mut nfc = false;
        fn walk_pre(
            p: &Json,
            regexes: &mut Vec<String>,
            byte_level: &mut bool,
            metaspace: &mut bool,
            prefix_space: &mut bool,
        ) -> Result<()> {
            match p["type"].as_str().unwrap_or("") {
                "Sequence" => {
                    for q in p["pretokenizers"].as_array().into_iter().flatten() {
                        walk_pre(q, regexes, byte_level, metaspace, prefix_space)?;
                    }
                }
                "Split" => {
                    if let Some(r) = p["pattern"]["Regex"].as_str() {
                        regexes.push(r.to_string());
                    } else if let Some(s) = p["pattern"]["String"].as_str() {
                        regexes.push(fancy_regex::escape(s).to_string());
                    }
                }
                "ByteLevel" => {
                    *byte_level = true;
                    if p["use_regex"].as_bool().unwrap_or(true) {
                        regexes.push(RE_GPT2_FULL.to_string());
                    }
                }
                "Digits" => {
                    regexes.push(if p["individual_digits"].as_bool().unwrap_or(false) {
                        r"\p{N}".to_string()
                    } else {
                        r"\p{N}+".to_string()
                    });
                }
                "Metaspace" => {
                    *metaspace = true;
                    let scheme = p["prepend_scheme"].as_str().unwrap_or("always");
                    *prefix_space = p["add_prefix_space"].as_bool().unwrap_or(true) && scheme != "never";
                }
                "" => {}
                other => bail!("unsupported pre-tokenizer '{other}' in tokenizer.json"),
            }
            Ok(())
        }
        if !j["pre_tokenizer"].is_null() {
            walk_pre(&j["pre_tokenizer"], &mut regexes, &mut byte_level, &mut metaspace, &mut prefix_space)?;
        }
        fn walk_norm(n: &Json, nfc: &mut bool, metaspace: &mut bool, prefix: &mut bool) {
            match n["type"].as_str().unwrap_or("") {
                "Sequence" => {
                    for m in n["normalizers"].as_array().into_iter().flatten() {
                        walk_norm(m, nfc, metaspace, prefix);
                    }
                }
                "NFC" => *nfc = true,
                "Prepend" => {
                    if n["prepend"].as_str() == Some("\u{2581}") {
                        *prefix = true;
                    }
                }
                "Replace" => {
                    if n["content"].as_str() == Some("\u{2581}") {
                        *metaspace = true;
                    }
                }
                _ => {}
            }
        }
        if !j["normalizer"].is_null() {
            walk_norm(&j["normalizer"], &mut nfc, &mut metaspace, &mut prefix_space);
        }
        let decoder_str = j["decoder"].to_string();
        if decoder_str.contains("\"ByteLevel\"") {
            byte_level = true;
        }
        let byte_fallback = model["byte_fallback"].as_bool().unwrap_or(false);
        let kind = if byte_level {
            TokenizerKind::ByteBpe
        } else if metaspace || byte_fallback || decoder_str.contains("ByteFallback") {
            TokenizerKind::SpmMerges
        } else {
            TokenizerKind::ByteBpe
        };
        if kind == TokenizerKind::SpmMerges {
            for (i, t) in tokens.iter().enumerate() {
                if types[i] == TOKEN_NORMAL && t.len() == 6 && t.starts_with("<0x") && t.ends_with('>') {
                    types[i] = TOKEN_BYTE;
                }
            }
            regexes.clear();
        }
        let pre_name = if kind == TokenizerKind::ByteBpe { guess_pre_name(&regexes) } else { "spm".into() };
        let mut tok = Self::build(kind, tokens, vec![], types, merges, regexes, pre_name)?;
        tok.ignore_merges = model["ignore_merges"].as_bool().unwrap_or(false);
        tok.add_space_prefix = kind == TokenizerKind::SpmMerges && prefix_space;
        tok.nfc = nfc;

        let token_str = |v: &Json| -> Option<String> {
            match v {
                Json::String(s) => Some(s.clone()),
                Json::Object(o) => o.get("content").and_then(|c| c.as_str()).map(String::from),
                _ => None,
            }
        };
        let id_of = |tok: &Tokenizer, v: &Json| token_str(v).and_then(|s| tok.token_to_id.get(&s).copied());
        let cfg_id = |k: &str| -> Option<u32> {
            match &model_cfg[k] {
                Json::Number(n) => n.as_u64().map(|x| x as u32),
                Json::Array(a) => a.first().and_then(|x| x.as_u64()).map(|x| x as u32),
                _ => None,
            }
        };
        tok.bos = id_of(&tok, &cfg["bos_token"]).or_else(|| cfg_id("bos_token_id"));
        tok.eos = id_of(&tok, &cfg["eos_token"]).or_else(|| cfg_id("eos_token_id"));
        tok.pad = id_of(&tok, &cfg["pad_token"]).or_else(|| cfg_id("pad_token_id"));
        tok.unk = id_of(&tok, &cfg["unk_token"]);
        // BOS: explicit flag, else whatever the post-processor prepends.
        let post = &j["post_processor"];
        let post_bos = post["single"].as_array().and_then(|seq| {
            let first = seq.first()?;
            let name = first["SpecialToken"]["id"].as_str()?;
            tok.token_to_id.get(name).copied()
        });
        tok.add_bos = match cfg["add_bos_token"].as_bool() {
            Some(b) => b,
            None => post_bos.is_some() || kind == TokenizerKind::SpmMerges,
        };
        if tok.bos.is_none() {
            tok.bos = post_bos;
        }
        tok.add_eos = cfg["add_eos_token"].as_bool().unwrap_or(false);
        if tok.add_bos && tok.bos.is_none() {
            tok.add_bos = false;
        }
        tok.chat_template = match &cfg["chat_template"] {
            Json::String(s) => Some(s.clone()),
            Json::Array(list) => list
                .iter()
                .find(|t| t["name"].as_str() == Some("default"))
                .or_else(|| list.first())
                .and_then(|t| t["template"].as_str())
                .map(String::from),
            _ => None,
        };
        Ok(tok)
    }

    pub fn info(&self) -> TokenizerInfo {
        TokenizerInfo {
            kind: self.kind,
            pre: self.pre_name.clone(),
            vocab_size: self.tokens.len(),
            merges: self.merges.len(),
            bos: self.bos.map(|i| self.token_text(i).to_string()),
            eos: self.eos.map(|i| self.token_text(i).to_string()),
            add_bos: self.add_bos,
            has_chat_template: self.chat_template.is_some(),
        }
    }

    pub fn kind(&self) -> TokenizerKind {
        self.kind
    }
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }
    pub fn token_types(&self) -> &[i32] {
        &self.token_types
    }
    pub fn merges_text(&self) -> &[String] {
        &self.merges_text
    }
    pub fn pre_name(&self) -> &str {
        &self.pre_name
    }
    pub fn add_space_prefix(&self) -> bool {
        self.add_space_prefix
    }
    pub fn token_text(&self, id: u32) -> &str {
        self.tokens.get(id as usize).map(|s| s.as_str()).unwrap_or("")
    }
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.token_to_id.get(text).copied()
    }
    pub fn is_control(&self, id: u32) -> bool {
        matches!(self.token_types.get(id as usize), Some(&TOKEN_CONTROL))
    }

    /// Tokens that should stop generation.
    pub fn end_of_generation_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = [self.eos, self.eot].into_iter().flatten().collect();
        for name in [
            "<|im_end|>",
            "<|eot_id|>",
            "<|eom_id|>",
            "<|end_of_text|>",
            "<|endoftext|>",
            "<end_of_turn>",
            "<|end|>",
            "</s>",
            "<|END_OF_TURN_TOKEN|>",
            "<|return|>",
        ] {
            if let Some(id) = self.token_id(name) {
                if self.is_control(id) || self.token_types[id as usize] == TOKEN_USER_DEFINED {
                    ids.push(id);
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Token used to pad batches (never trained on).
    pub fn pad_id(&self) -> u32 {
        self.pad.or(self.eos).or(self.unk).unwrap_or(0)
    }

    fn split_specials<'a>(&self, text: &'a str, parse_special: bool) -> Vec<Fragment<'a>> {
        let mut out = vec![];
        if self.specials.is_empty() {
            out.push(Fragment::Text(text));
            return out;
        }
        let mut last = 0;
        let mut pos = 0;
        let bytes = text.as_bytes();
        while pos < text.len() {
            // Only char boundaries can start a special token.
            if !text.is_char_boundary(pos) {
                pos += 1;
                continue;
            }
            let ch = text[pos..].chars().next().unwrap();
            let mut matched = None;
            if let Some(cands) = self.specials_by_first.get(&ch) {
                for &ci in cands {
                    let (s, id, always) = &self.specials[ci];
                    if (parse_special || *always) && bytes[pos..].starts_with(s.as_bytes()) {
                        matched = Some((s.len(), *id));
                        break;
                    }
                }
            }
            if let Some((len, id)) = matched {
                if pos > last {
                    out.push(Fragment::Text(&text[last..pos]));
                }
                out.push(Fragment::Token(id));
                pos += len;
                last = pos;
            } else {
                pos += ch.len_utf8();
            }
        }
        if last < text.len() {
            out.push(Fragment::Text(&text[last..]));
        }
        out
    }

    fn pre_split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut pieces = vec![text];
        for re in &self.regexes {
            let mut next = Vec::with_capacity(pieces.len() * 2);
            for piece in pieces {
                let mut last = 0;
                for m in re.find_iter(piece) {
                    let Ok(m) = m else { break };
                    if m.start() > last {
                        next.push(&piece[last..m.start()]);
                    }
                    if m.end() > m.start() {
                        next.push(&piece[m.start()..m.end()]);
                    }
                    last = m.end();
                }
                if last < piece.len() {
                    next.push(&piece[last..]);
                }
            }
            pieces = next;
        }
        pieces
    }

    fn push_byte_fallback(&self, piece: &str, out: &mut Vec<u32>) {
        for b in piece.bytes() {
            let id = match self.kind {
                TokenizerKind::ByteBpe => {
                    let c = self.byte_encoder[b as usize];
                    let mut buf = [0u8; 4];
                    self.token_to_id.get(c.encode_utf8(&mut buf) as &str).copied()
                }
                _ => self.byte_tokens[b as usize],
            };
            if let Some(id) = id.or(self.unk) {
                out.push(id);
            }
        }
    }

    /// Score-driven SentencePiece merging (llm_tokenizer_spm in llama.cpp).
    fn spm_scores(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let mut syms = init_symbols(text);
        let mut heap = BinaryHeap::new();
        let try_add = |heap: &mut BinaryHeap<SpmBigram>, syms: &[Sym], l: Option<usize>, r: Option<usize>| {
            let (Some(l), Some(r)) = (l, r) else { return };
            let s = &text[syms[l].start..syms[r].end];
            if let Some(&id) = self.token_to_id.get(s) {
                heap.push(SpmBigram { score: self.scores[id as usize], left: l, right: r, size: s.len() });
            }
        };
        for i in 1..syms.len() {
            try_add(&mut heap, &syms, Some(i - 1), Some(i));
        }
        while let Some(b) = heap.pop() {
            let (l, r) = (syms[b.left], syms[b.right]);
            let (ln, rn) = (l.end - l.start, r.end - r.start);
            if ln == 0 || rn == 0 || ln + rn != b.size || l.next != Some(b.right) {
                continue;
            }
            syms[b.left].end = r.end;
            syms[b.left].next = r.next;
            syms[b.right].start = syms[b.right].end; // mark dead
            if let Some(n) = r.next {
                syms[n].prev = Some(b.left);
            }
            try_add(&mut heap, &syms, syms[b.left].prev, Some(b.left));
            try_add(&mut heap, &syms, Some(b.left), syms[b.left].next);
        }
        let mut cur = Some(0);
        while let Some(i) = cur {
            let s = &text[syms[i].start..syms[i].end];
            if let Some(&id) = self.token_to_id.get(s) {
                out.push(id);
            } else {
                self.push_byte_fallback(s, out);
            }
            cur = syms[i].next;
        }
    }

    /// Rank-driven BPE merging (llm_tokenizer_bpe in llama.cpp).
    fn bpe_merge(&self, word: &str, out: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }
        if self.ignore_merges {
            if let Some(&id) = self.token_to_id.get(word) {
                out.push(id);
                return;
            }
        }
        let mut syms = init_symbols(word);
        for s in syms.iter_mut() {
            s.id = self.token_to_id.get(&word[s.start..s.end]).copied();
        }
        let mut heap = BinaryHeap::new();
        let try_add = |heap: &mut BinaryHeap<BpeBigram>, syms: &[Sym], l: Option<usize>, r: Option<usize>| {
            let (Some(l), Some(r)) = (l, r) else { return };
            let (Some(li), Some(ri)) = (syms[l].id, syms[r].id) else { return };
            if let Some(&(rank, _)) = self.merges.get(&(li, ri)) {
                heap.push(BpeBigram { rank, left: l, right: r, left_id: li, right_id: ri });
            }
        };
        for i in 1..syms.len() {
            try_add(&mut heap, &syms, Some(i - 1), Some(i));
        }
        while let Some(b) = heap.pop() {
            let (l, r) = (syms[b.left], syms[b.right]);
            if l.start == l.end
                || r.start == r.end
                || l.next != Some(b.right)
                || l.id != Some(b.left_id)
                || r.id != Some(b.right_id)
            {
                continue;
            }
            let merged = self.merges[&(b.left_id, b.right_id)].1;
            syms[b.left].end = r.end;
            syms[b.left].next = r.next;
            syms[b.left].id = Some(merged);
            syms[b.right].start = syms[b.right].end;
            syms[b.right].id = None;
            if let Some(n) = r.next {
                syms[n].prev = Some(b.left);
            }
            try_add(&mut heap, &syms, syms[b.left].prev, Some(b.left));
            try_add(&mut heap, &syms, Some(b.left), syms[b.left].next);
        }
        let mut cur = Some(0);
        while let Some(i) = cur {
            match syms[i].id {
                Some(id) => out.push(id),
                None => self.push_byte_fallback(&word[syms[i].start..syms[i].end], out),
            }
            cur = syms[i].next;
        }
    }

    fn encode_bpe_word(&self, piece: &str, out: &mut Vec<u32>) {
        let word: String = if self.kind == TokenizerKind::ByteBpe {
            piece.bytes().map(|b| self.byte_encoder[b as usize]).collect()
        } else {
            piece.to_string()
        };
        if word.len() <= 64 {
            if let Some(ids) = self.cache.lock().unwrap().get(&word) {
                out.extend_from_slice(ids);
                return;
            }
        }
        let mut ids = Vec::new();
        self.bpe_merge(&word, &mut ids);
        if word.len() <= 64 {
            let mut cache = self.cache.lock().unwrap();
            if cache.len() > 200_000 {
                cache.clear();
            }
            cache.insert(word, ids.clone());
        }
        out.extend_from_slice(&ids);
    }

    /// Encode text into token ids.
    ///
    /// * `add_special` adds BOS/EOS when the model asks for it.
    /// * `parse_special` turns strings such as `<|im_start|>` into their control tokens.
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let normalized;
        let text = if self.nfc {
            use unicode_normalization::UnicodeNormalization;
            normalized = text.nfc().collect::<String>();
            normalized.as_str()
        } else {
            text
        };
        let mut out = Vec::with_capacity(text.len() / 3 + 2);
        if add_special && self.add_bos {
            if let Some(b) = self.bos {
                out.push(b);
            }
        }
        let mut prev_special = true;
        for frag in self.split_specials(text, parse_special) {
            match frag {
                Fragment::Token(id) => {
                    out.push(id);
                    prev_special = true;
                }
                Fragment::Text(t) => {
                    match self.kind {
                        TokenizerKind::SpmScores | TokenizerKind::SpmMerges => {
                            let mut s = String::with_capacity(t.len() + 3);
                            if self.add_space_prefix && prev_special {
                                s.push(SPACE_ESCAPE);
                            }
                            for c in t.chars() {
                                s.push(if c == ' ' { SPACE_ESCAPE } else { c });
                            }
                            if self.kind == TokenizerKind::SpmScores {
                                self.spm_scores(&s, &mut out);
                            } else {
                                self.bpe_merge(&s, &mut out);
                            }
                        }
                        TokenizerKind::ByteBpe => {
                            for piece in self.pre_split(t) {
                                self.encode_bpe_word(piece, &mut out);
                            }
                        }
                    }
                    prev_special = false;
                }
            }
        }
        if add_special && self.add_eos {
            if let Some(e) = self.eos {
                out.push(e);
            }
        }
        out
    }

    /// Raw bytes for a single token.
    pub fn token_bytes(&self, id: u32, skip_special: bool) -> Vec<u8> {
        let Some(text) = self.tokens.get(id as usize) else { return vec![] };
        match self.token_types[id as usize] {
            TOKEN_CONTROL | TOKEN_UNKNOWN | TOKEN_UNUSED => {
                if skip_special {
                    vec![]
                } else {
                    text.as_bytes().to_vec()
                }
            }
            TOKEN_USER_DEFINED => text.as_bytes().to_vec(),
            TOKEN_BYTE => {
                let hex = text.trim_start_matches("<0x").trim_end_matches('>');
                u8::from_str_radix(hex, 16).map(|b| vec![b]).unwrap_or_default()
            }
            _ => match self.kind {
                TokenizerKind::ByteBpe => {
                    let mut v = Vec::with_capacity(text.len());
                    for c in text.chars() {
                        match self.byte_decoder.get(&c) {
                            Some(b) => v.push(*b),
                            None => {
                                let mut buf = [0u8; 4];
                                v.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                            }
                        }
                    }
                    v
                }
                _ => text.replace(SPACE_ESCAPE, " ").into_bytes(),
            },
        }
    }

    pub fn decode_bytes(&self, ids: &[u32], skip_special: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            out.extend(self.token_bytes(id, skip_special));
        }
        if self.add_space_prefix && out.first() == Some(&b' ') {
            out.remove(0);
        }
        out
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, skip_special)).into_owned()
    }
}

/// Best effort mapping from a list of pre-tokenizer regexes to a llama.cpp `tokenizer.ggml.pre` name.
fn guess_pre_name(regexes: &[String]) -> String {
    let joined = regexes.join("\u{0}");
    let llama3_orig = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let qwen2_orig = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    if joined == llama3_orig || joined == RE_LLAMA3 {
        "llama-bpe".into()
    } else if joined == qwen2_orig || joined == RE_QWEN2 {
        "qwen2".into()
    } else if regexes.len() == 2 && regexes[0] == r"\p{N}" {
        "smollm".into()
    } else if joined == RE_GPT2_FULL || joined == RE_GPT2 {
        "gpt-2".into()
    } else {
        "default".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_bpe() -> Tokenizer {
        // Byte-level BPE vocabulary: all bytes + a few merges.
        let enc = bytes_to_unicode();
        let mut tokens: Vec<String> = enc.iter().map(|c| c.to_string()).collect();
        let sp = enc[b' ' as usize];
        let extra = [format!("{sp}h"), "he".into(), "ll".into(), "hell".into(), "hello".into(), format!("{sp}hello")];
        tokens.extend(extra.iter().cloned());
        tokens.push("<|end|>".into());
        let mut types = vec![TOKEN_NORMAL; tokens.len()];
        *types.last_mut().unwrap() = TOKEN_CONTROL;
        let merges = vec![
            "h e".to_string(),
            "l l".into(),
            "he ll".into(),
            "hell o".into(),
            format!("{sp} hello"),
        ];
        Tokenizer::build(TokenizerKind::ByteBpe, tokens, vec![], types, merges, vec![RE_GPT2.into()], "gpt-2".into())
            .unwrap()
    }

    #[test]
    fn bpe_roundtrip_and_merges() {
        let t = toy_bpe();
        let ids = t.encode("hello hello", false, false);
        assert_eq!(t.token_text(ids[0]), "hello");
        assert_eq!(ids.len(), 2);
        assert_eq!(t.decode(&ids, false), "hello hello");
        let ids = t.encode("héllo<|end|>", false, true);
        assert_eq!(*ids.last().unwrap(), t.token_id("<|end|>").unwrap());
        assert_eq!(t.decode(&ids, true), "héllo");
        // Without special parsing the control string is split into bytes.
        let ids = t.encode("<|end|>", false, false);
        assert!(ids.len() > 1);
    }

    #[test]
    fn spm_scores_merge_and_byte_fallback() {
        let mut tokens = vec!["<unk>".to_string(), "<s>".into(), "</s>".into()];
        let mut types = vec![TOKEN_UNKNOWN, TOKEN_CONTROL, TOKEN_CONTROL];
        for b in 0..256 {
            tokens.push(format!("<0x{b:02X}>"));
            types.push(TOKEN_BYTE);
        }
        for (s, _) in [("▁", 0), ("h", 0), ("i", 0), ("hi", 0), ("▁hi", 0), ("!", 0)] {
            tokens.push(s.to_string());
            types.push(TOKEN_NORMAL);
        }
        let mut scores = vec![0.0; tokens.len()];
        let hi = tokens.iter().position(|t| t == "hi").unwrap();
        let shi = tokens.iter().position(|t| t == "▁hi").unwrap();
        scores[hi] = -1.0;
        scores[shi] = -0.5;
        let mut t = Tokenizer::build(TokenizerKind::SpmScores, tokens, scores, types, vec![], vec![], "spm".into()).unwrap();
        t.add_space_prefix = true;
        t.bos = Some(1);
        t.add_bos = true;
        let ids = t.encode("hi hi!é", true, false);
        assert_eq!(ids[0], 1);
        assert_eq!(t.token_text(ids[1]), "▁hi");
        assert_eq!(t.token_text(ids[2]), "▁hi");
        assert_eq!(t.token_text(ids[3]), "!");
        // é is not in the vocabulary -> two UTF-8 byte tokens.
        assert_eq!(ids.len(), 6);
        assert_eq!(t.decode(&ids, true), "hi hi!é");
    }
}
