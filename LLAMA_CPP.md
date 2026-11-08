# Local inference with llama.cpp

The local launcher uses Claude Code for slash commands, subagents, and tool permissions. Inference goes to your `llama-server` through its [Anthropic-compatible Messages API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#anthropic-compatible-api-endpoints). Install a current [Claude Code](https://code.claude.com/docs/en/setup) and Python 3.10+. The launcher needs no Anthropic API key or Claude subscription.

## Start the model

If you already run `llama-server`, reuse it. It needs `/v1/messages`, a tool-capable GGUF chat template, and `--jinja`. To start a new server on the same machine:

```bash
llama-server -m /path/to/model.gguf --alias job-search-local \
  --host 127.0.0.1 --port 8080 --jinja --ctx-size 65536 --parallel 1
```

Add the GPU flags you normally use for your hardware. Keep enough context for the posting, profile, drafts, and tool results. Start with 65,536 tokens if your model and hardware can support it. With multiple parallel slots, check the effective context **per slot** in the server log. A single slot can queue calls from reviewer and scoring agents.

Try [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) first if you already serve it. To use DeepSeek Flash, select a GGUF that loads on your llama.cpp build and has a working tool template. [DeepSeek V4.1 Flash support](https://github.com/ggml-org/llama.cpp/pull/31086) landed on October 25, 2026, and some [V4.1 Flash 1017 GGUF variants](https://github.com/ggml-org/llama.cpp/issues/31385) have different build requirements. The launcher uses whichever model your server exposes.

## Check and launch

From the repository root:

```bash
python3 tools/llama_cpp.py --check
python3 tools/llama_cpp.py
```

`--check` tests the Messages API and whether the model's chat template accepts a Claude Code reminder, without starting inference or launching Claude Code. The default URL is `http://127.0.0.1:8080`. Use the server base URL, without `/v1/messages`. If the server lists several models, select the exact model ID or alias with `--model`.

```bash
python3 tools/llama_cpp.py --server http://127.0.0.1:8080 --model job-search-local
```

Set `LLAMA_CPP_URL` and `LLAMA_CPP_MODEL` in your shell to reuse those choices. If you set an API key on `llama-server`, set `LLAMA_CPP_API_KEY` in the client shell. For a separate inference machine, use its address on your trusted LAN or VPN and configure a server key. Keep the key and your profile files out of a public GitHub fork.

Use `--context-tokens` if automatic context detection cannot find the effective per-slot value. You can also set Claude Code's [`CLAUDE_CODE_MAX_CONTEXT_TOKENS`](https://code.claude.com/docs/en/model-config#correct-the-window-for-a-gateway-or-custom-model-id). Pass additional Claude Code arguments after `--`:

```bash
python3 tools/llama_cpp.py --context-tokens 65536 -- --resume
```

On Windows, use `python tools\llama_cpp.py` if `python3` is unavailable. Inside Claude Code, run `/setup`, then `/scrape`, `/rank`, or `/apply`. For a first smoke test, ask the model to read the root `AGENTS.md` and report its heading, then try `/setup` with sample profile text. If the model prints tool markup or loops without running tools, check `--jinja`, the model's chat template, and your llama.cpp version.

## Qwen3.8 system-message error

If `--check` reports **System message must be at the beginning**, the model's template rejects system reminders that Claude Code can insert later in a conversation. Generate a patched copy of the template your server is already using:

```bash
python3 tools/patch_qwen38_template.py --url http://127.0.0.1:8080 \
  --output /path/to/qwen38-claude-code.jinja
```

For a router serving multiple models, add `--model` with the Qwen model's ID. If you already have the template file, use `--input /path/to/template.jinja` in place of `--url`. Restart the Qwen server with its existing flags plus `--chat-template-file /path/to/qwen38-claude-code.jinja`, then rerun `--check`. The patch leaves the GGUF intact and preserves the template's other behavior.

## What still needs a network or human review

- The launcher reduces Claude Code's optional network traffic. Job board searches, posting fetches, and company research still reach public websites. If built-in web tools are unavailable, `/scrape` can use the installed portal CLIs, and `/apply` accepts pasted posting text or a directly supplied URL via the repository fetch helper. Review company claims against sources before using them.
- `/gmail-sync` and `/notion-sync` still depend on their external connectors and accounts. The local inference endpoint supplies model responses.
- A text-only model can work with pasted CV text and profile interviews. Extract text from PDFs with `pdftotext` or `pypdf` and use OCR for scanned documents. For generated CVs and cover letters, install Poppler so the workflow can measure PDF layout. Open both compiled PDFs yourself to complete the visual inspection if the model cannot read rendered pages.
