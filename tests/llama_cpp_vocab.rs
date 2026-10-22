//! Compares the tokenizer against llama.cpp's reference tokenizations.
//!
//! Point `LLAMA_VOCAB_DIR` at llama.cpp's `models/` folder (it contains
//! `ggml-vocab-*.gguf` plus `.inp` / `.out` files) and run
//! `cargo test --release --test llama_cpp_vocab -- --nocapture`.
//! Without the variable the test is skipped.

use std::path::Path;

use candle_core::quantized::gguf_file;
use llm_finetune::tokenizer::Tokenizer;

fn run_vocab(path: &Path) -> Result<(usize, usize, Vec<String>), String> {
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let content = gguf_file::Content::read(&mut f).map_err(|e| e.to_string())?;
    let tok = Tokenizer::from_gguf(&content.metadata).map_err(|e| e.to_string())?;
    let inp = std::fs::read(path.with_extension("gguf.inp")).map_err(|e| e.to_string())?;
    let inp = String::from_utf8_lossy(&inp).to_string();
    let out =
        std::fs::read_to_string(path.with_extension("gguf.out")).map_err(|e| e.to_string())?;
    let cases: Vec<&str> = inp.split("\n__ggml_vocab_test__\n").collect();
    let expected: Vec<&str> = out.lines().collect();
    let mut ok = 0;
    let mut failures = vec![];
    for (i, text) in cases.iter().enumerate() {
        let Some(exp) = expected.get(i) else { break };
        let exp: Vec<u32> = exp.split_whitespace().map(|x| x.parse().unwrap()).collect();
        let got = tok.encode(text, false, false);
        if got == exp {
            ok += 1;
        } else if failures.len() < 3 {
            failures.push(format!(
                "  text {:?}\n    expected {:?}\n    got      {:?}",
                text, exp, got
            ));
        }
    }
    Ok((ok, cases.len().min(expected.len()), failures))
}

#[test]
fn llama_cpp_reference_tokenizations() {
    let Ok(dir) = std::env::var("LLAMA_VOCAB_DIR") else {
        eprintln!("LLAMA_VOCAB_DIR not set; skipping");
        return;
    };
    // Vocabularies this project promises to handle exactly.
    let required = [
        "llama-spm",
        "llama-bpe",
        "qwen2",
        "qwen35",
        "gpt-2",
        "phi-3",
        "starcoder",
        "refact",
        "falcon",
        "mpt",
        "command-r",
        "deepseek-coder",
        "deepseek-llm",
    ];
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    entries.sort();
    let mut hard_fail = vec![];
    for p in entries {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if !name.ends_with(".gguf") || !p.with_extension("gguf.out").exists() {
            continue;
        }
        let short = name
            .trim_start_matches("ggml-vocab-")
            .trim_end_matches(".gguf")
            .to_string();
        match run_vocab(&p) {
            Ok((ok, total, failures)) => {
                println!("{short:<16} {ok}/{total}");
                for f in &failures {
                    println!("{f}");
                }
                if required.contains(&short.as_str()) && ok != total {
                    hard_fail.push(short);
                }
            }
            Err(e) => {
                println!("{short:<16} unsupported: {e}");
                if required.contains(&short.as_str()) {
                    hard_fail.push(short);
                }
            }
        }
    }
    assert!(
        hard_fail.is_empty(),
        "tokenizer mismatches for: {hard_fail:?}"
    );
}

/// Renders every chat template shipped in llama.cpp's `models/templates`.
#[test]
fn llama_cpp_chat_templates() {
    use llm_finetune::chat::{tokenize_rendered, ChatFormatter, Conversation, TemplateKind};
    let Ok(dir) = std::env::var("LLAMA_VOCAB_DIR") else {
        eprintln!("LLAMA_VOCAB_DIR not set; skipping");
        return;
    };
    let dir = Path::new(&dir);
    let mut f = std::fs::File::open(dir.join("ggml-vocab-qwen2.gguf")).unwrap();
    let content = gguf_file::Content::read(&mut f).unwrap();
    let mut tok = Tokenizer::from_gguf(&content.metadata).unwrap();
    let conv = Conversation {
        system: Some("You are helpful.".into()),
        turns: vec![
            ("Hi there".into(), Some("Hello! How can I help?".into())),
            ("What is 2+2?".into(), Some("4.".into())),
        ],
    };
    let mut paths: Vec<_> = std::fs::read_dir(dir.join("templates"))
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    paths.retain(|p| p.extension().map(|e| e == "jinja").unwrap_or(false));
    paths.sort();
    let mut ok = 0;
    for p in &paths {
        tok.chat_template = Some(std::fs::read_to_string(p).unwrap());
        let fmt = ChatFormatter::new(TemplateKind::Model, &tok);
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        if fmt.kind() != TemplateKind::Model {
            println!(
                "fallback {name}: {}",
                fmt.jinja_error.clone().unwrap_or_default()
            );
            continue;
        }
        let r = fmt.render(&conv, false).unwrap();
        assert_eq!(r.train_spans.len(), 2, "{name}");
        for (&(s, e), answer) in r.train_spans.iter().zip(["Hello! How can I help?", "4."]) {
            assert!(
                r.text[s..e].starts_with(answer),
                "{name}: span {:?}",
                &r.text[s..e]
            );
        }
        let (_, mask) = tokenize_rendered(&tok, &fmt, &r, false);
        assert!(mask.iter().any(|m| *m), "{name}: nothing trained");
        ok += 1;
    }
    println!(
        "{ok}/{} templates render with the model's own Jinja",
        paths.len()
    );
    assert!(
        ok * 10 >= paths.len() * 9,
        "too many templates fail: {ok}/{}",
        paths.len()
    );
}
