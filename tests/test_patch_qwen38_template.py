import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from tools import patch_qwen38_template


STOCK_TEMPLATE = r"""{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- for message in messages %}
    {%- set content = message.content|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>user\n' + content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
"""


class TestQwenTemplatePatch(unittest.TestCase):
    def test_keeps_reminder_in_place_and_preserves_the_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stock.jinja"
            target = Path(directory) / "patched.jinja"
            stock = STOCK_TEMPLATE.replace("\n", "\r\n").encode()
            source.write_bytes(stock)

            result = patch_qwen38_template.main(["--input", str(source), "--output", str(target)])

            self.assertEqual(result, 0)
            self.assertEqual(source.read_bytes(), stock)
            self.assertEqual(
                target.read_bytes(),
                stock.replace(
                    b"raise_exception('System message must be at the beginning.')",
                    rb"'<|im_start|>system\n' + content + '<|im_end|>\n'",
                ),
            )

    def test_rejects_unknown_template_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stock.jinja"
            target = Path(directory) / "patched.jinja"
            source.write_text("other template", encoding="utf-8")
            error = io.StringIO()

            with redirect_stderr(error), self.assertRaises(SystemExit) as result:
                patch_qwen38_template.main(["--input", str(source), "--output", str(target)])

            self.assertEqual(result.exception.code, 1)
            self.assertIn("Expected one Qwen system-message guard", error.getvalue())
            self.assertFalse(target.exists())

    def test_reads_props_for_the_selected_model_with_bearer_auth(self):
        response = io.StringIO(json.dumps({"chat_template": STOCK_TEMPLATE}))
        with patch.dict("os.environ", {"LLAMA_CPP_API_KEY": "test-token"}), patch.object(
            patch_qwen38_template, "urlopen", return_value=response,
        ) as urlopen:
            template = patch_qwen38_template.server_template(
                "http://localhost:8080/v1/", "model/with spaces",
            )

        self.assertEqual(template, STOCK_TEMPLATE)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:8080/props?model=model%2Fwith+spaces")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
