import contextlib
import io
import json
import os
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import patch

from tools import llama_cpp


@contextlib.contextmanager
def llama_server(models=("qwen-local",), template_error=False):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode("utf-8"))

        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization"), None))
            if self.path == "/v1/models":
                self.respond(200, {"data": [{"id": model} for model in models]})
            elif self.path.startswith("/props?"):
                self.respond(200, {"default_generation_settings": {"n_ctx": 65536}})
            else:
                self.respond(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            if self.path != "/v1/messages/count_tokens":
                self.respond(404, {})
            elif template_error:
                self.respond(400, {"error": {"message": "System message must be at the beginning."}})
            else:
                self.respond(200, {"input_tokens": 67})

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            worker.join()


class LlamaCppTests(unittest.TestCase):
    def run_launcher(self, args, env=None):
        with patch.dict(os.environ, env or {}, clear=True), contextlib.redirect_stderr(io.StringIO()) as stderr:
            status = llama_cpp.main(args)
        return status, stderr.getvalue()

    def test_check_discovers_model_and_context_with_local_bearer_auth(self):
        with llama_server() as (url, requests):
            status, output = self.run_launcher(["--server", url + "/v1/", "--check"], {"LLAMA_CPP_API_KEY": "test-key"})
        self.assertEqual(status, 0, output)
        self.assertIn("65,536 tokens", output)
        self.assertEqual([path for path, _, _ in requests], [
            "/v1/models", "/v1/messages/count_tokens", "/props?model=qwen-local",
        ])
        self.assertTrue(all(auth == "Bearer test-key" for _, auth, _ in requests))
        body = requests[1][2]
        self.assertEqual(body["model"], "qwen-local")
        self.assertEqual([message["role"] for message in body["messages"]], ["user", "system"])
        self.assertIn("input_schema", body["tools"][0])

    def test_router_requires_a_model_when_multiple_are_advertised(self):
        with llama_server(("qwen-local", "deepseek-local")) as (url, requests):
            status, output = self.run_launcher(["--server", url, "--check"])
        self.assertEqual(status, 1)
        self.assertIn("qwen-local, deepseek-local", output)
        self.assertEqual([path for path, _, _ in requests], ["/v1/models"])

    def test_qwen_template_failure_explains_fix(self):
        with llama_server(template_error=True) as (url, _):
            status, output = self.run_launcher(["--server", url, "--model", "qwen-local", "--check"])
        self.assertEqual(status, 1)
        self.assertIn("Qwen chat-template fix", output)

    def test_launches_repo_workflows_with_only_local_model_and_credentials(self):
        with llama_server() as (url, _), patch.object(llama_cpp.shutil, "which", return_value="claude"), \
                patch.object(llama_cpp.subprocess, "call", return_value=7) as run:
            status, _ = self.run_launcher(["--", "-p", "hello"], {
                "LLAMA_CPP_URL": url,
                "LLAMA_CPP_MODEL": "qwen-local",
                "LLAMA_CPP_API_KEY": "local-key",
                "ANTHROPIC_API_KEY": "hosted-key",
                "ANTHROPIC_AUTH_TOKEN": "hosted-token",
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "32768",
            })
        self.assertEqual(status, 7)
        self.assertEqual(run.call_args.args[0], ["claude", "--model", "qwen-local", "-p", "hello"])
        self.assertEqual(run.call_args.kwargs["cwd"], llama_cpp.ROOT)
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["ANTHROPIC_BASE_URL"], url)
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "local-key")
        self.assertEqual(env["CLAUDE_CODE_SUBAGENT_MODEL"], "qwen-local")
        self.assertEqual(env["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"], "1")
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "32768")
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("CLAUDE_CODE_USE_BEDROCK", env)
        self.assertNotIn("LLAMA_CPP_API_KEY", env)


if __name__ == "__main__":
    unittest.main()
