"""
Pure-function unit tests for backend.sandbox.Sandbox.

These tests exercise only the static/pure helpers on the Sandbox class:
    * Sandbox.extract_code
    * Sandbox.parse_multi_file_manifest
    * Sandbox._detect_language

None of these methods spawn subprocesses, load models, or touch the network,
so the whole file runs in well under a second and is safe to run in CI.
"""
import os
import sys
import pytest

# Make the repo root importable regardless of where pytest is invoked from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.sandbox import Sandbox  # noqa: E402


# ---------------------------------------------------------------------------
# Sandbox.extract_code
# ---------------------------------------------------------------------------
class TestExtractCode:
    def test_extracts_python_fenced_block(self):
        text = "Here is the code:\n```python\nprint('hi')\n```\nThanks!"
        assert Sandbox.extract_code(text).strip() == "print('hi')"

    def test_extracts_unlabeled_fenced_block(self):
        text = "```\nx = 1\ny = 2\n```"
        out = Sandbox.extract_code(text).strip()
        assert "x = 1" in out and "y = 2" in out

    def test_prefers_language_labeled_over_bare_fence(self):
        text = (
            "First a note:\n"
            "```\njust text\n```\n"
            "Now the code:\n"
            "```python\nprint('real')\n```\n"
        )
        out = Sandbox.extract_code(text)
        assert "print('real')" in out

    def test_returns_original_when_no_fence(self):
        # extract_code should degrade gracefully — returning either the raw
        # text or an empty string, but must NEVER raise.
        text = "print('no fences here')"
        result = Sandbox.extract_code(text)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Sandbox.parse_multi_file_manifest
# ---------------------------------------------------------------------------
class TestParseMultiFileManifest:
    def test_parses_two_files(self):
        text = (
            '<file path="index.html">\n'
            "<!DOCTYPE html><html><body>Hi</body></html>\n"
            "</file>\n"
            '<file path="app.js">\n'
            "console.log('ok');\n"
            "</file>\n"
        )
        files = Sandbox.parse_multi_file_manifest(text)
        assert isinstance(files, dict)
        assert "index.html" in files
        assert "app.js" in files
        assert "<!DOCTYPE html>" in files["index.html"]
        assert "console.log" in files["app.js"]

    def test_returns_falsy_on_plain_text(self):
        result = Sandbox.parse_multi_file_manifest("just some prose, no tags")
        # Implementation may return {} or None; both are acceptable "no manifest".
        assert not result

    def test_ignores_malformed_tag(self):
        text = "<file>missing path attr</file>"
        result = Sandbox.parse_multi_file_manifest(text)
        assert not result or "" not in result


# ---------------------------------------------------------------------------
# Sandbox._detect_language
# ---------------------------------------------------------------------------
class TestDetectLanguage:
    def setup_method(self):
        self.sb = Sandbox(timeout=5)

    def test_detects_python(self):
        code = "import os\ndef f():\n    return 1\nprint(f())"
        lang = self.sb._detect_language(code)
        assert lang == "python"

    def test_detects_javascript(self):
        code = "const x = 1;\nconsole.log(x);"
        lang = self.sb._detect_language(code)
        assert lang in ("javascript", "typescript")

    def test_html_falls_back_to_python(self):
        # HTML is not in LANG_SIGNATURES (single-file execute() handles only
        # executable languages; HTML goes through execute_workspace instead).
        # With no strong signal for any supported language, _detect_language
        # is documented to fall back to 'python'.
        code = "<!DOCTYPE html>\n<html><body><h1>hi</h1></body></html>"
        lang = self.sb._detect_language(code)
        assert lang == "python"

    def test_detects_bash(self):
        code = "#!/bin/bash\nset -euo pipefail\nfor f in *.txt; do\n  echo \"$f\"\ndone"
        lang = self.sb._detect_language(code)
        assert lang == "bash"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
