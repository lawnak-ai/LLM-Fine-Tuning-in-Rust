//! Reads uploaded files (pdf, docx, odt, xlsx, csv/tsv, xml, json/jsonl, html, text)
//! into plain text, table rows or conversations.

use std::collections::HashMap;
use std::io::{Cursor, Read};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value as Json;

use crate::chat::Conversation;

#[derive(Debug, Clone)]
pub enum Extracted {
    /// Free text documents (used for continued pre-training).
    Text(Vec<String>),
    /// Tabular records (column names + rows).
    Records {
        columns: Vec<String>,
        rows: Vec<Vec<String>>,
    },
    /// Ready-made chat conversations.
    Conversations(Vec<Conversation>),
}

pub fn file_kind(name: &str) -> String {
    std::path::Path::new(name)
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_else(|| "txt".to_string())
}

pub fn extract(name: &str, bytes: &[u8]) -> Result<Extracted> {
    let kind = file_kind(name);
    match kind.as_str() {
        "pdf" => Ok(Extracted::Text(vec![pdf_text(bytes)?])),
        "docx" | "docm" | "dotx" => Ok(Extracted::Text(vec![docx_text(bytes)?])),
        "odt" => Ok(Extracted::Text(vec![odt_text(bytes)?])),
        "doc" => bail!("old binary .doc files are not supported; save the file as .docx (Word: File > Save As > Word Document)"),
        "xlsx" | "xlsm" => xlsx_records(bytes),
        "csv" => csv_records(&decode_text(bytes), None),
        "tsv" | "tab" => csv_records(&decode_text(bytes), Some(b'\t')),
        "xml" => xml_extract(&decode_text(bytes)),
        "json" => json_extract(&decode_text(bytes)),
        "jsonl" | "ndjson" => jsonl_extract(&decode_text(bytes)),
        "html" | "htm" | "xhtml" => Ok(Extracted::Text(vec![clean_text(&html_to_text(&decode_text(bytes)))])),
        _ => {
            if looks_binary(bytes) {
                bail!("'{name}' looks like a binary file of an unsupported type (.{kind})");
            }
            Ok(Extracted::Text(vec![clean_text(&decode_text(bytes))]))
        }
    }
}

fn looks_binary(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(4096)];
    head.iter().filter(|&&b| b == 0).count() > head.len() / 32
        && !(head.starts_with(&[0xFF, 0xFE]) || head.starts_with(&[0xFE, 0xFF]))
}

/// Decode UTF-8 (with or without BOM) or UTF-16 with BOM; fall back to lossy UTF-8.
pub fn decode_text(bytes: &[u8]) -> String {
    if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8_lossy(rest).into_owned();
    }
    let utf16 = |be: bool| -> String {
        let units: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|c| {
                if be {
                    u16::from_be_bytes([c[0], c[1]])
                } else {
                    u16::from_le_bytes([c[0], c[1]])
                }
            })
            .collect();
        String::from_utf16_lossy(&units)
    };
    if bytes.starts_with(&[0xFF, 0xFE]) {
        return utf16(false);
    }
    if bytes.starts_with(&[0xFE, 0xFF]) {
        return utf16(true);
    }
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        // Most likely Latin-1 / Windows-1252.
        Err(_) => bytes.iter().map(|&b| b as char).collect(),
    }
}

/// Normalise whitespace while keeping paragraph structure.
pub fn clean_text(s: &str) -> String {
    let s = s
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\u{0c}', "\n\n")
        .replace('\u{a0}', " ");
    let mut out = String::with_capacity(s.len());
    let mut blank = 0;
    for line in s.lines() {
        let line = line.trim_end();
        if line.trim().is_empty() {
            blank += 1;
            continue;
        }
        if !out.is_empty() {
            out.push_str(if blank > 0 { "\n\n" } else { "\n" });
        }
        blank = 0;
        out.push_str(line);
    }
    out
}

fn pdf_text(bytes: &[u8]) -> Result<String> {
    let owned = bytes.to_vec();
    // pdf-extract may panic on malformed files; contain it.
    let res = std::panic::catch_unwind(move || pdf_extract::extract_text_from_mem(&owned));
    let text = match res {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => bail!("could not read PDF: {e}"),
        Err(_) => bail!("could not read PDF (the file is damaged or uses unsupported features)"),
    };
    let text = clean_text(&fix_pdf_hyphenation(&text));
    if text.trim().is_empty() {
        bail!("the PDF contains no extractable text (scanned images need OCR first)");
    }
    Ok(text)
}

/// Join words split by a hyphen at the end of a line ("trans-\nformer" -> "transformer").
fn fix_pdf_hyphenation(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '-'
            && i > 0
            && chars[i - 1].is_alphabetic()
            && chars.get(i + 1) == Some(&'\n')
            && chars.get(i + 2).map(|c| c.is_lowercase()).unwrap_or(false)
        {
            i += 2;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn zip_file(bytes: &[u8], inner: &str) -> Result<String> {
    let mut zip =
        zip::ZipArchive::new(Cursor::new(bytes)).context("not a valid zip-based document")?;
    let mut f = zip
        .by_name(inner)
        .with_context(|| format!("{inner} not found in document"))?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}

fn xml_doc(s: &str) -> Result<roxmltree::Document<'_>> {
    let opts = roxmltree::ParsingOptions {
        allow_dtd: true,
        ..Default::default()
    };
    roxmltree::Document::parse_with_options(s, opts).map_err(|e| anyhow!("invalid XML: {e}"))
}

fn docx_text(bytes: &[u8]) -> Result<String> {
    let xml = zip_file(bytes, "word/document.xml")?;
    let doc = xml_doc(&xml)?;
    let mut out = String::new();
    if let Some(body) = doc.descendants().find(|n| n.tag_name().name() == "body") {
        docx_walk(body, &mut out);
    }
    let text = clean_text(&out);
    if text.trim().is_empty() {
        bail!("the Word document contains no text");
    }
    Ok(text)
}

fn docx_para(p: roxmltree::Node, out: &mut String) {
    for d in p.descendants() {
        match d.tag_name().name() {
            "t" => out.push_str(d.text().unwrap_or("")),
            "tab" => out.push('\t'),
            "br" | "cr" => out.push('\n'),
            _ => {}
        }
    }
}

fn docx_walk(node: roxmltree::Node, out: &mut String) {
    for c in node.children().filter(|c| c.is_element()) {
        match c.tag_name().name() {
            "p" => {
                docx_para(c, out);
                out.push('\n');
            }
            "tbl" => {
                for tr in c.children().filter(|n| n.tag_name().name() == "tr") {
                    let cells: Vec<String> = tr
                        .children()
                        .filter(|n| n.tag_name().name() == "tc")
                        .map(|tc| {
                            let mut s = String::new();
                            for p in tc.descendants().filter(|n| n.tag_name().name() == "p") {
                                if !s.is_empty() {
                                    s.push(' ');
                                }
                                docx_para(p, &mut s);
                            }
                            s.trim().to_string()
                        })
                        .collect();
                    out.push_str(&cells.join(" | "));
                    out.push('\n');
                }
                out.push('\n');
            }
            "sectPr" => {}
            _ => docx_walk(c, out),
        }
    }
}

fn odt_text(bytes: &[u8]) -> Result<String> {
    let xml = zip_file(bytes, "content.xml")?;
    let doc = xml_doc(&xml)?;
    let mut out = String::new();
    for n in doc
        .descendants()
        .filter(|n| matches!(n.tag_name().name(), "p" | "h"))
    {
        // Skip paragraphs nested in other paragraphs (they are visited through the parent).
        if n.ancestors()
            .skip(1)
            .any(|a| matches!(a.tag_name().name(), "p" | "h"))
        {
            continue;
        }
        for d in n.descendants() {
            if d.is_text() {
                out.push_str(d.text().unwrap_or(""));
            } else {
                match d.tag_name().name() {
                    "s" => out.push(' '),
                    "tab" => out.push('\t'),
                    "line-break" => out.push('\n'),
                    _ => {}
                }
            }
        }
        out.push('\n');
    }
    Ok(clean_text(&out))
}

fn col_index(cell_ref: &str) -> usize {
    let mut n = 0usize;
    for c in cell_ref.chars().take_while(|c| c.is_ascii_alphabetic()) {
        n = n * 26 + (c.to_ascii_uppercase() as usize - 'A' as usize + 1);
    }
    n.saturating_sub(1)
}

fn xlsx_records(bytes: &[u8]) -> Result<Extracted> {
    let shared: Vec<String> = match zip_file(bytes, "xl/sharedStrings.xml") {
        Ok(xml) => {
            let doc = xml_doc(&xml)?;
            doc.descendants()
                .filter(|n| n.tag_name().name() == "si")
                .map(|si| {
                    si.descendants()
                        .filter(|t| t.tag_name().name() == "t")
                        .filter_map(|t| t.text())
                        .collect::<String>()
                })
                .collect()
        }
        Err(_) => vec![],
    };
    let sheet = zip_file(bytes, "xl/worksheets/sheet1.xml")?;
    let doc = xml_doc(&sheet)?;
    let mut rows: Vec<Vec<String>> = vec![];
    for row in doc.descendants().filter(|n| n.tag_name().name() == "row") {
        let mut cells: Vec<String> = vec![];
        for (i, c) in row
            .children()
            .filter(|n| n.tag_name().name() == "c")
            .enumerate()
        {
            let idx = c.attribute("r").map(col_index).unwrap_or(i);
            let t = c.attribute("t").unwrap_or("");
            let v = c
                .children()
                .find(|n| n.tag_name().name() == "v")
                .and_then(|n| n.text())
                .unwrap_or("");
            let val = match t {
                "s" => v
                    .parse::<usize>()
                    .ok()
                    .and_then(|i| shared.get(i).cloned())
                    .unwrap_or_default(),
                "inlineStr" => c
                    .descendants()
                    .filter(|n| n.tag_name().name() == "t")
                    .filter_map(|n| n.text())
                    .collect(),
                _ => v.to_string(),
            };
            if cells.len() <= idx {
                cells.resize(idx + 1, String::new());
            }
            cells[idx] = val;
        }
        if cells.iter().any(|c| !c.trim().is_empty()) {
            rows.push(cells);
        }
    }
    if rows.is_empty() {
        bail!("the spreadsheet is empty");
    }
    let header = rows.remove(0);
    Ok(make_records(header, rows))
}

fn make_records(header: Vec<String>, mut rows: Vec<Vec<String>>) -> Extracted {
    let width = rows
        .iter()
        .map(|r| r.len())
        .max()
        .unwrap_or(0)
        .max(header.len());
    let mut columns: Vec<String> = Vec::with_capacity(width);
    for i in 0..width {
        let name = header
            .get(i)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("column{}", i + 1));
        let mut unique = name.clone();
        let mut k = 2;
        while columns.contains(&unique) {
            unique = format!("{name}_{k}");
            k += 1;
        }
        columns.push(unique);
    }
    for r in rows.iter_mut() {
        r.resize(width, String::new());
    }
    Extracted::Records { columns, rows }
}

fn csv_records(text: &str, delim: Option<u8>) -> Result<Extracted> {
    let delim = delim.unwrap_or_else(|| {
        let first = text.lines().next().unwrap_or("");
        b",;\t|"
            .iter()
            .copied()
            .max_by_key(|d| first.bytes().filter(|b| b == d).count())
            .unwrap_or(b',')
    });
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delim)
        .flexible(true)
        .has_headers(true)
        .from_reader(text.as_bytes());
    let header: Vec<String> = rdr.headers()?.iter().map(|s| s.to_string()).collect();
    let mut rows = vec![];
    for rec in rdr.records() {
        let rec = rec.context("malformed CSV row")?;
        let row: Vec<String> = rec.iter().map(|s| s.to_string()).collect();
        if row.iter().any(|c| !c.trim().is_empty()) {
            rows.push(row);
        }
    }
    if rows.is_empty() {
        bail!("the CSV file has a header but no rows");
    }
    Ok(make_records(header, rows))
}

fn xml_extract(text: &str) -> Result<Extracted> {
    let doc = xml_doc(text)?;
    // A "record" is an element whose element children are all leaves, or that has attributes only.
    let mut counts: HashMap<String, usize> = HashMap::new();
    for n in doc.descendants().filter(|n| n.is_element()) {
        let kids: Vec<_> = n.children().filter(|c| c.is_element()).collect();
        let leafy = !kids.is_empty() && kids.iter().all(|k| !k.children().any(|g| g.is_element()));
        let attr_only = kids.is_empty() && n.attributes().len() >= 2;
        if leafy || attr_only {
            *counts.entry(n.tag_name().name().to_string()).or_insert(0) += 1;
        }
    }
    let best = counts
        .into_iter()
        .filter(|(_, c)| *c >= 2)
        .max_by_key(|(_, c)| *c);
    if let Some((tag, _)) = best {
        let mut columns: Vec<String> = vec![];
        let mut maps: Vec<HashMap<String, String>> = vec![];
        for n in doc
            .descendants()
            .filter(|n| n.is_element() && n.tag_name().name() == tag)
        {
            let mut m = HashMap::new();
            for a in n.attributes() {
                m.insert(a.name().to_string(), a.value().to_string());
            }
            for k in n.children().filter(|c| c.is_element()) {
                let v: String = k
                    .descendants()
                    .filter(|d| d.is_text())
                    .filter_map(|d| d.text())
                    .collect();
                m.insert(k.tag_name().name().to_string(), v.trim().to_string());
            }
            for key in m.keys() {
                if !columns.contains(key) {
                    columns.push(key.clone());
                }
            }
            maps.push(m);
        }
        let rows = maps
            .into_iter()
            .map(|m| {
                columns
                    .iter()
                    .map(|c| m.get(c).cloned().unwrap_or_default())
                    .collect()
            })
            .collect();
        return Ok(Extracted::Records { columns, rows });
    }
    // Document-like XML: keep the text.
    let mut out = String::new();
    for t in doc.descendants().filter(|n| n.is_text()) {
        let s = t.text().unwrap_or("").trim();
        if !s.is_empty() {
            out.push_str(s);
            out.push('\n');
        }
    }
    Ok(Extracted::Text(vec![clean_text(&out)]))
}

fn json_to_string(v: &Json) -> String {
    match v {
        Json::String(s) => s.clone(),
        Json::Null => String::new(),
        other => other.to_string(),
    }
}

/// OpenAI / ShareGPT style message lists -> conversation.
fn conversation_from(obj: &serde_json::Map<String, Json>) -> Option<Conversation> {
    let msgs = obj
        .get("messages")
        .or_else(|| obj.get("conversations"))
        .or_else(|| obj.get("conversation"))?
        .as_array()?;
    let mut conv = Conversation::default();
    let mut pending_user: Option<String> = None;
    for m in msgs {
        let role = m
            .get("role")
            .or_else(|| m.get("from"))
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let content = m
            .get("content")
            .or_else(|| m.get("value"))
            .or_else(|| m.get("text"))
            .map(json_to_string)
            .unwrap_or_default();
        match role.as_str() {
            "system" => conv.system = Some(content),
            "user" | "human" | "prompter" => {
                if let Some(u) = pending_user.take() {
                    conv.turns.push((u, None));
                }
                pending_user = Some(content);
            }
            "assistant" | "gpt" | "bot" | "model" => {
                let u = pending_user.take().unwrap_or_default();
                conv.turns.push((u, Some(content)));
            }
            _ => {}
        }
    }
    conv.turns.retain(|(_, a)| a.is_some());
    if conv.turns.is_empty() {
        None
    } else {
        Some(conv)
    }
}

fn objects_to_extracted(items: Vec<Json>) -> Result<Extracted> {
    if items.is_empty() {
        bail!("no records found");
    }
    if items.iter().all(|v| v.is_string()) {
        return Ok(Extracted::Text(items.iter().map(json_to_string).collect()));
    }
    let objs: Vec<&serde_json::Map<String, Json>> =
        items.iter().filter_map(|v| v.as_object()).collect();
    if objs.is_empty() {
        bail!("expected a list of objects or strings");
    }
    let convs: Vec<Conversation> = objs.iter().filter_map(|o| conversation_from(o)).collect();
    if convs.len() * 2 >= objs.len() && !convs.is_empty() {
        return Ok(Extracted::Conversations(convs));
    }
    let mut columns: Vec<String> = vec![];
    for o in &objs {
        for k in o.keys() {
            if !columns.contains(k) {
                columns.push(k.clone());
            }
        }
    }
    let rows = objs
        .iter()
        .map(|o| {
            columns
                .iter()
                .map(|c| o.get(c).map(json_to_string).unwrap_or_default())
                .collect()
        })
        .collect();
    Ok(Extracted::Records { columns, rows })
}

fn json_extract(text: &str) -> Result<Extracted> {
    let v: Json = match serde_json::from_str(text) {
        Ok(v) => v,
        // Many ".json" datasets are really JSON lines.
        Err(e) => return jsonl_extract(text).map_err(|_| anyhow!("invalid JSON: {e}")),
    };
    match v {
        Json::Array(items) => objects_to_extracted(items),
        Json::Object(o) => {
            if let Some(c) = conversation_from(&o) {
                return Ok(Extracted::Conversations(vec![c]));
            }
            // {"data": [...]} or similar wrapper.
            if let Some(arr) = o
                .values()
                .filter_map(|v| v.as_array())
                .max_by_key(|a| a.len())
            {
                if !arr.is_empty() {
                    return objects_to_extracted(arr.clone());
                }
            }
            if let Some(t) = o.get("text").and_then(|t| t.as_str()) {
                return Ok(Extracted::Text(vec![t.to_string()]));
            }
            objects_to_extracted(vec![Json::Object(o)])
        }
        Json::String(s) => Ok(Extracted::Text(vec![s])),
        _ => bail!("unsupported JSON content"),
    }
}

fn jsonl_extract(text: &str) -> Result<Extracted> {
    let mut items = vec![];
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        items.push(
            serde_json::from_str::<Json>(line)
                .with_context(|| format!("invalid JSON on line {}", i + 1))?,
        );
    }
    objects_to_extracted(items)
}

/// Very small HTML to text converter (drops scripts/styles, keeps block structure).
pub fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let lower = html.to_ascii_lowercase();
    let mut i = 0;
    let bytes = html.as_bytes();
    while i < html.len() {
        if bytes[i] == b'<' {
            let end = match html[i..].find('>') {
                Some(e) => i + e,
                None => break,
            };
            let tag = lower[i + 1..end]
                .trim_start_matches('/')
                .split(|c: char| c.is_whitespace() || c == '/')
                .next()
                .unwrap_or("")
                .to_string();
            if (tag == "script" || tag == "style") && !lower[i + 1..end].starts_with('/') {
                let close = format!("</{tag}");
                i = lower[end..]
                    .find(&close)
                    .map(|p| end + p)
                    .unwrap_or(html.len());
                continue;
            }
            if matches!(
                tag.as_str(),
                "p" | "div"
                    | "br"
                    | "li"
                    | "tr"
                    | "h1"
                    | "h2"
                    | "h3"
                    | "h4"
                    | "h5"
                    | "h6"
                    | "section"
                    | "article"
                    | "table"
                    | "ul"
                    | "ol"
                    | "pre"
                    | "blockquote"
            ) {
                out.push('\n');
            }
            if tag == "td" || tag == "th" {
                out.push(' ');
            }
            i = end + 1;
        } else if bytes[i] == b'&' {
            let end = html[i..].find(';').filter(|e| *e <= 10).map(|e| i + e);
            match end {
                Some(e) => {
                    let ent = &html[i + 1..e];
                    let rep = match ent {
                        "amp" => Some('&'),
                        "lt" => Some('<'),
                        "gt" => Some('>'),
                        "quot" => Some('"'),
                        "apos" | "#39" => Some('\''),
                        "nbsp" => Some(' '),
                        _ if ent.starts_with("#x") => u32::from_str_radix(&ent[2..], 16)
                            .ok()
                            .and_then(char::from_u32),
                        _ if ent.starts_with('#') => {
                            ent[1..].parse::<u32>().ok().and_then(char::from_u32)
                        }
                        _ => None,
                    };
                    match rep {
                        Some(c) => {
                            out.push(c);
                            i = e + 1;
                        }
                        None => {
                            out.push('&');
                            i += 1;
                        }
                    }
                }
                None => {
                    out.push('&');
                    i += 1;
                }
            }
        } else {
            let ch = html[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_detects_delimiter() {
        let e = extract(
            "a.csv",
            b"question;answer\nWhat is 2+2?;4\nCapital of France?;Paris\n",
        )
        .unwrap();
        match e {
            Extracted::Records { columns, rows } => {
                assert_eq!(columns, vec!["question", "answer"]);
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[1][1], "Paris");
            }
            _ => panic!("expected records"),
        }
    }

    #[test]
    fn xml_records_and_text() {
        let xml = r#"<data><item><q>Hi</q><a>Hello</a></item><item><q>Bye</q><a>Goodbye</a></item></data>"#;
        match extract("a.xml", xml.as_bytes()).unwrap() {
            Extracted::Records { columns, rows } => {
                assert!(columns.contains(&"q".to_string()));
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("expected records"),
        }
        let xml = r#"<doc><title>Rust</title><body>Rust is fast.</body></doc>"#;
        match extract("b.xml", xml.as_bytes()).unwrap() {
            Extracted::Text(t) => assert!(t[0].contains("Rust is fast.")),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn json_messages_become_conversations() {
        let j = r#"[{"messages":[{"role":"system","content":"S"},{"role":"user","content":"U"},{"role":"assistant","content":"A"}]}]"#;
        match extract("a.json", j.as_bytes()).unwrap() {
            Extracted::Conversations(c) => {
                assert_eq!(c[0].system.as_deref(), Some("S"));
                assert_eq!(c[0].turns[0], ("U".to_string(), Some("A".to_string())));
            }
            _ => panic!("expected conversations"),
        }
        let j = "{\"instruction\":\"I\",\"output\":\"O\"}\n{\"instruction\":\"I2\",\"output\":\"O2\"}\n";
        match extract("a.jsonl", j.as_bytes()).unwrap() {
            Extracted::Records { rows, .. } => assert_eq!(rows.len(), 2),
            _ => panic!("expected records"),
        }
    }

    #[test]
    fn html_and_cleanup() {
        let t = html_to_text("<html><style>x{}</style><p>Hello&nbsp;<b>world</b> &amp; you</p><script>bad()</script></html>");
        assert_eq!(clean_text(&t), "Hello world & you");
        assert_eq!(fix_pdf_hyphenation("trans-\nformer"), "transformer");
    }
}
