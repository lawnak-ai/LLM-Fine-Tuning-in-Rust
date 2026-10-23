# LLM Fine-Tuning in Rust

A small web app for fine-tuning open LLMs on your own documents and Q&A data, written in Rust.
It trains LoRA adapters on top of a frozen model, on the CPU or a GPU.

What it can do:

- Load GGUF models (the format used by llama.cpp, Ollama and LM Studio) or Hugging Face folders (`config.json` + `*.safetensors`).
- Read training data from PDF, Word (.docx), Excel (.xlsx), CSV/TSV, XML, JSON/JSONL, .odt, HTML, Markdown and plain text.
- Train on CPU, NVIDIA (CUDA) or Apple (Metal). Quantized GGUF models stay quantized while training (QLoRA), so they need much less memory.
- Save the result as a LoRA adapter (llama.cpp GGUF or PEFT safetensors) or as a merged model (GGUF with any common quantization, or safetensors).
- Chat with the model in the browser before and after training.

![Training screen](docs/train.png)

## Download

Ready-made builds for Windows, Linux and macOS (Apple Silicon) are on the
[Releases](https://github.com/lawnak-ai/LLM-Fine-Tuning-in-Rust/releases) page.
Unzip, run `llm-finetune.exe` (Windows) or `llm-finetune`, and the browser opens by itself.

- Windows: if SmartScreen shows "Windows protected your PC", click "More info" and "Run anyway".
- macOS: the first time, right-click the file and choose Open.
- The Windows and Linux builds run on the CPU (AVX2 needed). For an NVIDIA GPU, build from source as below.

## Building from source

Install Rust from https://rustup.rs, then build:

```bash
cargo build --release                    # CPU
cargo build --release --features cuda    # NVIDIA GPU, needs the CUDA toolkit (nvcc)
cargo build --release --features metal   # Apple Silicon GPU
```

Put a model in `workspace/models/` (or upload it from the page) and start the server:

```bash
./target/release/llm-finetune
```

Then open http://127.0.0.1:7860.

Some small models that are good for a first try (all on Hugging Face):

- `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, file `qwen2.5-0.5b-instruct-q8_0.gguf` (0.5 GB)
- `HuggingFaceTB/SmolLM2-360M-Instruct` (safetensors folder, 0.7 GB)
- `bartowski/Llama-3.2-1B-Instruct-GGUF`, the Q8_0 file (1.3 GB)
- `Qwen/Qwen3-1.7B-GGUF`, the Q8_0 file (1.8 GB)

Models downloaded with Ollama also work, they are plain GGUF files in `~/.ollama/models/blobs/` (the biggest `sha256-...` file of the model).

## Using the UI

The page has five tabs:

1. **Model & device**: pick a model, choose Auto / CPU / NVIDIA / Apple GPU and how the base weights are kept in memory, then load it.
2. **Data**: drag files in. Each file shows what was found (texts or Q&A pairs). For tables you can choose which columns are the prompt and the answer, and "Preview" shows the formatted example with the trained part highlighted.
3. **Train**: pick a preset or set epochs, learning rate, LoRA rank etc. You get a live loss chart, speed and time left. "Stop & save" keeps what was trained so far.
4. **Test**: chat with the model. The adapter is used right after training.
5. **Export**: save or load adapters, export a merged model, download files.

![Data screen](docs/data.png)

Everything goes into `workspace/` (`models/` and `outputs/run-<date>/`). See `llm-finetune serve --help` for the port and folders. By default the server only listens on 127.0.0.1.

## Training data

- PDF, Word, ODT, TXT, Markdown and HTML are used as plain text, so the model learns the content. The text is joined and cut into chunks of "max tokens per example".
- CSV, TSV, Excel, XML and JSON with columns like `question`/`answer`, `instruction`/`input`/`output` or `prompt`/`response` become chat examples. They are put in the model's chat format and only the answer is trained. The columns can be changed in the UI.
- JSON/JSONL with `messages` (OpenAI style) or `conversations` (ShareGPT) are used as multi-turn chats.
- Other tables are used as text, one row per example.

Old `.doc` files and scanned PDFs without a text layer don't work, save as `.docx` or run OCR first.
There is an example file for each format in `examples/data`.

## Command line

```bash
# train
llm-finetune train --model qwen2.5-0.5b-instruct-q8_0.gguf \
    --data examples/data/faq.csv examples/data/manual.pdf \
    --epochs 3 --lr 2e-4 --rank 16 --output out/my-run \
    --merge-gguf q8_0        # optional, also writes out/my-run/merged.gguf

# try it
llm-finetune generate --model qwen2.5-0.5b-instruct-q8_0.gguf --adapter out/my-run \
    --prompt "What are your opening hours?"

# merge later
llm-finetune merge --model base.gguf --adapter out/my-run --output merged-q4.gguf --quant q4_k

# model info
llm-finetune info --model base.gguf
```

`--help` on each command lists all options. `train --config file.json` takes the same settings as the UI.

## Supported models

- Llama architecture: Llama 2, Llama 3.x, TinyLlama, SmolLM, Mistral 7B and fine-tunes of these
- Qwen 2 / 2.5
- Qwen 3 (dense)

GGUF types F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and Q2_K to Q6_K are supported (not the IQ types).
The chat format is taken from the model's own template when it can be rendered, otherwise one of the built-in formats is used (ChatML, Llama 3, Llama 2, Mistral, Zephyr, Phi-3, Alpaca, plain).

## Memory and speed

"Base weights in memory" on the first tab:

- keep as in file: a quantized GGUF stays quantized, uses the least memory
- f32: fastest on CPU, 4 bytes per parameter
- f16 / bf16: half of that
- q8_0 / q4_k: quantize a safetensors model while loading

Gradient checkpointing (under Advanced, on by default) saves a lot of memory for a bit of speed.

On a 4-core cloud VM with a 0.5B Qwen-sized model I get about 25-32 tokens/s with f32 weights (2.3 GB RAM) and about 20 tokens/s with q8_0/q4_0 (1.9 GB). 1000 Q&A pairs of ~100 tokens is roughly an hour per epoch there. More cores or a GPU are a lot faster.

The release build uses `target-cpu=native` (`.cargo/config.toml`), so the binary is built for the CPU of the machine that compiled it.

## Using the output

A training run writes to `workspace/outputs/run-<date>/`:

- `adapter.gguf`: for llama.cpp (`llama-cli -m base.gguf --lora adapter.gguf`, also llama-server) or Ollama (`ADAPTER ./adapter.gguf` in a Modelfile, with `FROM` the same base model)
- `adapter_model.safetensors` + `adapter_config.json`: Hugging Face PEFT (`PeftModel.from_pretrained(model, "run-dir")`)
- `training_log.json`: settings and loss per step

A merged `.gguf` from the Export tab is a normal model file that works in Ollama (`FROM ./merged.gguf`), LM Studio, llama.cpp etc.

## Code layout

```
src/
  tokenizer.rs    SentencePiece and byte-level BPE, from GGUF or tokenizer.json
  model/          transformer, GGUF/safetensors loading, LoRA layers, fused ops
  data/           file readers and sample building
  chat.rs         chat templates (minijinja) and answer masks
  train.rs        training loop
  export.rs       adapters and merged models
  gguf_writer.rs  GGUF writer
  generate.rs     sampling
  server.rs       HTTP API, the page is src/ui/index.html
  main.rs         CLI
```

Tensors and the CPU/CUDA/Metal kernels come from [candle](https://github.com/huggingface/candle).

## Tests

```bash
cargo test --release
```

There are unit tests for the readers, tokenizer, ops and export code, a gradient check against finite differences, and an end-to-end test that trains on a CSV and reloads the saved adapters.

If you have a llama.cpp checkout, `LLAMA_VOCAB_DIR=<llama.cpp>/models cargo test --release --test llama_cpp_vocab` compares the tokenizer with llama.cpp's reference outputs and renders its collection of real chat templates.
`examples/dump_logits.rs` and `examples/bench_train.rs` are small tools for comparing outputs with llama.cpp and measuring speed.

## Limitations

- Only Llama and Qwen style dense models for now (no Gemma, Phi, MoE or vision models).
- One device per run.
- I have only tested on CPU so far. CUDA and Metal come from candle and should work, but haven't been tried yet.
- YaRN / dynamic RoPE scaling is ignored, which is fine as long as training sequences fit in the original context.

## License

MIT
