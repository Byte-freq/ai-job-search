#!/usr/bin/env python3
"""Allow Claude Code's system reminders in a Qwen3.8 llama-server chat template."""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


SYSTEM_GUARD = re.compile(
    r"""^(?P<opening>[ \t]*{{-?[ \t]*)"""
    r"""raise_exception\((?P<quote>['"])System message must be at the beginning\.(?P=quote)\)"""
    r"""(?P<closing>[ \t]*-?}}[ \t]*)(?=\r?$)""",
    re.MULTILINE,
)


def patch_template(template: str) -> str:
    # Keep the reminder where Claude Code sent it so later turns can reuse the cached prefix.
    # The template has already rendered the message into `content` at this point.
    reminder = r"'<|im_start|>system\n' + content + '<|im_end|>\n'"
    patched, count = SYSTEM_GUARD.subn(
        lambda match: match["opening"] + reminder + match["closing"],
        template,
    )
    if count != 1 or "<|im_start|>" not in template:
        raise ValueError("Expected one Qwen system-message guard. The template may already support system reminders.")
    return patched


def server_template(base_url: str, model: str | None) -> str:
    url = urlsplit(base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username is not None
        or url.query
        or url.fragment
    ):
        raise ValueError("Use the llama-server HTTP(S) base URL without credentials, a query, or a fragment.")

    path = url.path.rstrip("/").removesuffix("/v1") + "/props"
    query = urlencode({"model": model}) if model else ""
    request = Request(urlunsplit((url.scheme, url.netloc, path, query, "")))
    if api_key := os.environ.get("LLAMA_CPP_API_KEY"):
        request.add_unredirected_header("Authorization", f"Bearer {api_key}")
    with urlopen(request, timeout=30) as response:
        props = json.load(response)

    template = props.get("chat_template") if isinstance(props, dict) else None
    if not isinstance(template, str) or not template:
        raise ValueError("llama-server /props did not return a chat template. For a model router, pass --model.")
    return template


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="llama-server base URL (uses LLAMA_CPP_API_KEY if set)")
    source.add_argument("--input", type=Path, help="existing Jinja template file")
    parser.add_argument("--model", help="model ID when using llama-server in router mode")
    parser.add_argument("--output", type=Path, required=True, help="new Jinja template file")
    args = parser.parse_args(argv)
    if args.model and args.input:
        parser.error("--model requires --url")

    try:
        if args.input:
            template = args.input.read_bytes().decode("utf-8")
        else:
            template = server_template(args.url, args.model)
        patched = patch_template(template)
        with args.output.open("x", encoding="utf-8", newline="") as output:
            output.write(patched)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
