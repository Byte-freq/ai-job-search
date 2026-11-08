#!/usr/bin/env python3
"""Run this project's Claude Code workflows against a llama.cpp server."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = "http://127.0.0.1:8080"
CONTEXT_ENV = "CLAUDE_CODE_MAX_CONTEXT_TOKENS"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def server_url(value):
    url = urlsplit(value.strip())
    if (
        url.scheme not in {"http", "https"} or not url.hostname
        or url.username is not None or url.query or url.fragment
    ):
        raise ValueError("Use an http(s) llama-server URL without credentials, query, or fragment.")
    return url.geturl().rstrip("/").removesuffix("/v1")


def request_json(server, path, key, payload=None):
    headers = {"Accept": "application/json"}
    body = None
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        headers["anthropic-version"] = "2023-06-01"
        body = json.dumps(payload).encode("utf-8")
    request = Request(server + path, data=body, headers=headers)
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=60) as response:
            result = json.load(response)
    except HTTPError as error:
        message = error.read(8192).decode("utf-8", errors="replace")
        if "System message must be at the beginning" in message:
            hint = " Apply the Qwen chat-template fix in LLAMA_CPP.md, then restart llama-server."
        elif error.code in {401, 403}:
            hint = " Check LLAMA_CPP_API_KEY."
        elif error.code == 404 and path == "/v1/messages/count_tokens":
            hint = " Use a current llama-server build with the Anthropic Messages API."
        elif error.code == 503:
            hint = " Wait for the model to load, then try again."
        else:
            hint = " Check the llama-server log and confirm it was started with --jinja."
        raise RuntimeError(f"{path}: HTTP {error.code}.{hint}") from error
    except (URLError, OSError, ValueError) as error:
        raise RuntimeError(f"{path}: could not read a JSON response from {server} ({error}).") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"{path}: expected a JSON object from llama-server.")
    return result


def discover_model(server, key):
    listing = request_json(server, "/v1/models", key).get("data", [])
    models = list(dict.fromkeys(
        entry["id"] for entry in listing
        if isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry["id"]
    )) if isinstance(listing, list) else []
    if len(models) != 1:
        available = ", ".join(models) if models else "none"
        raise RuntimeError(f"Specify --model or LLAMA_CPP_MODEL. Server model IDs: {available}.")
    return models[0]


def check_messages(server, model, key):
    # Claude Code can inject an agent/skill reminder after the first user turn.
    result = request_json(server, "/v1/messages/count_tokens", key, {
        "model": model,
        "system": "You are a job search assistant.",
        "messages": [
            {"role": "user", "content": "Check the connection."},
            {"role": "system", "content": "A local test tool is available."},
        ],
        "tools": [{
            "name": "connection_check",
            "description": "A test tool with no side effects.",
            "input_schema": {"type": "object", "properties": {}},
        }],
    })
    if type(result.get("input_tokens")) is not int or result["input_tokens"] <= 0:
        raise RuntimeError("The Anthropic token-count endpoint did not return a positive input_tokens value.")


def discover_context(server, model, key):
    try:
        props = request_json(server, "/props?" + urlencode({"model": model}), key)
    except RuntimeError:
        return None
    settings = props.get("default_generation_settings")
    if isinstance(settings, dict):
        context = settings.get("n_ctx")
        if type(context) is int and context > 0:
            return context
    return None


def claude_environment(server, model, key, context):
    env = os.environ.copy()
    # Keep hosted credentials and provider choices out of this invocation.
    for name in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_CUSTOM_HEADERS", "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_MANTLE", "LLAMA_CPP_API_KEY",
    ):
        env.pop(name, None)
    env.update({
        "ANTHROPIC_BASE_URL": server,
        "ANTHROPIC_AUTH_TOKEN": key or "local",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL_FORCE": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_EFFORT_LEVEL": "auto",
        "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1",
        "ENABLE_TOOL_SEARCH": "false",
    })
    # The API needs room for workflow prompts as well as generated drafts.
    if context:
        env[CONTEXT_ENV] = str(context)
        env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", str(min(8192, max(1, context // 4))))
    host = urlsplit(server).hostname
    bypass = ",".join(filter(None, (env.get("NO_PROXY"), env.get("no_proxy"), host)))
    env["NO_PROXY"] = env["no_proxy"] = bypass
    return env


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    claude_args = []
    if "--" in argv:
        split = argv.index("--")
        argv, claude_args = argv[:split], argv[split + 1:]

    parser = argparse.ArgumentParser(description=__doc__, epilog="Pass Claude Code arguments after --.")
    parser.add_argument("--server", default=os.getenv("LLAMA_CPP_URL") or DEFAULT_SERVER,
                        help="llama-server base URL (env: LLAMA_CPP_URL, default: %(default)s)")
    parser.add_argument("--model", default=os.getenv("LLAMA_CPP_MODEL"),
                        help="served model ID (env: LLAMA_CPP_MODEL, autodetected on single-model servers)")
    parser.add_argument("--context-tokens", type=int, default=os.getenv(CONTEXT_ENV) or None,
                        help="effective server context window (default: read llama-server /props)")
    parser.add_argument("--check", action="store_true", help="validate the server and template without launching Claude Code")
    args = parser.parse_args(argv)
    if args.context_tokens is not None and args.context_tokens <= 0:
        parser.error("--context-tokens must be positive")
    if args.check and claude_args:
        parser.error("--check does not launch Claude Code; remove arguments after --")
    if any(arg == "--model" or arg.startswith("--model=") for arg in claude_args):
        parser.error("set the model before -- so the launcher can check it and route subagents")

    try:
        server = server_url(args.server)
        key = os.getenv("LLAMA_CPP_API_KEY")
        model = args.model or discover_model(server, key)
        check_messages(server, model, key)
        context = args.context_tokens or discover_context(server, model, key)
        context_label = f"{context:,} tokens" if context else "unknown; use --context-tokens to set it"
        print(f"llama.cpp ready: {server} | model: {model} | context: {context_label}", file=sys.stderr, flush=True)
        if args.check:
            return 0
        claude = shutil.which("claude")
        if not claude:
            raise RuntimeError("Claude Code CLI not found on PATH. Install it using the link in LLAMA_CPP.md.")
        env = claude_environment(server, model, key, context)
        return subprocess.call([claude, "--model", model, *claude_args], cwd=ROOT, env=env)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"llama.cpp setup: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
