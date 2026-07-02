"""
Unit tests for pure text/patch helpers on backend.orchestrator.AgentOrchestrator.

The methods under test do NOT touch models, disk, or network — they operate
purely on strings. To avoid triggering AgentOrchestrator.__init__ (which
loads psutil/torch/llama_cpp and may fail in a bare CI env), we bind the
unbound methods to a SimpleNamespace and call them directly.

Covered methods:
    * _clean_cutoff_notes  — strips training-cutoff disclaimers
    * _strip_thinking      — removes <think>...</think> reasoning blocks
    * _apply_search_replace_patch — applies Aider-style SEARCH/REPLACE patches
"""
import os
import sys
from types import SimpleNamespace

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import the class lazily and skip cleanly if heavy deps are missing
# (e.g. torch not installed on a lightweight CI runner).
try:
    from backend.orchestrator import AgentOrchestrator  # noqa: E402
except Exception as e:  # pragma: no cover - import-time env issue
    pytest.skip(
        f"Cannot import AgentOrchestrator ({e!r}); skipping orchestrator tests.",
        allow_module_level=True,
    )


def _bind(method_name):
    """Return AgentOrchestrator.<method_name> as an unbound function so we
    can call it with a fake `self` (SimpleNamespace) — no __init__ needed."""
    return getattr(AgentOrchestrator, method_name)


_FAKE_SELF = SimpleNamespace()


# ---------------------------------------------------------------------------
# _clean_cutoff_notes
# ---------------------------------------------------------------------------
class TestCleanCutoffNotes:
    fn = staticmethod(lambda text: _bind("_clean_cutoff_notes")(_FAKE_SELF, text))

    def test_empty_input_returns_input(self):
        assert self.fn("") == ""
        assert self.fn(None) is None

    def test_removes_parenthesized_cutoff_note(self):
        text = (
            "The capital of France is Paris. "
            "(Note: my training data only goes up to 2023.)"
        )
        out = self.fn(text)
        assert "Paris" in out
        assert "training data" not in out.lower()

    def test_removes_verify_official_sources_disclaimer(self):
        text = "The answer is 42. Always verify with official sources for the most up-to-date information."
        out = self.fn(text)
        assert "42" in out
        assert "verify with official sources" not in out.lower()

    def test_leaves_normal_text_intact(self):
        text = "Python is a great language for data science."
        assert self.fn(text) == text


# ---------------------------------------------------------------------------
# _strip_thinking
# ---------------------------------------------------------------------------
class TestStripThinking:
    fn = staticmethod(lambda text: _bind("_strip_thinking")(_FAKE_SELF, text))

    def test_removes_closed_think_block(self):
        text = "<think>let me reason...</think>The answer is 4."
        out = self.fn(text)
        assert "<think>" not in out
        assert "The answer is 4." in out

    def test_returns_content_when_entire_answer_is_in_think(self):
        text = "<think>The answer is 4.</think>"
        out = self.fn(text)
        assert "The answer is 4." in out
        assert "<think>" not in out and "</think>" not in out

    def test_no_tags_returns_stripped_text(self):
        text = "  Just a plain answer.  "
        assert self.fn(text) == "Just a plain answer."

    def test_unclosed_think_prefers_content_before_tag(self):
        text = "Here is the final answer: 42.\n<think>wait, is it really 42? let me redo..."
        out = self.fn(text)
        assert "42" in out
        # The self-questioning tail should not leak into the final output
        assert "let me redo" not in out

    def test_empty_input(self):
        assert self.fn("") == ""
        assert self.fn(None) is None


# ---------------------------------------------------------------------------
# _apply_search_replace_patch
# ---------------------------------------------------------------------------
class TestApplySearchReplacePatch:
    fn = staticmethod(
        lambda code, patch: _bind("_apply_search_replace_patch")(_FAKE_SELF, code, patch)
    )

    def test_returns_none_when_no_patch_markers(self):
        assert self.fn("x = 1", "just some prose") is None
        assert self.fn("x = 1", "") is None

    def test_applies_single_block(self):
        code = "def add(a, b):\n    return a - b\n"
        patch = (
            "<<<<<<< SEARCH\n"
            "    return a - b\n"
            "=======\n"
            "    return a + b\n"
            ">>>>>>> REPLACE\n"
        )
        out = self.fn(code, patch)
        assert out is not None
        assert "return a + b" in out
        assert "return a - b" not in out

    def test_applies_multiple_blocks(self):
        code = "a = 1\nb = 2\nc = 3\n"
        patch = (
            "<<<<<<< SEARCH\n"
            "a = 1\n"
            "=======\n"
            "a = 10\n"
            ">>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\n"
            "c = 3\n"
            "=======\n"
            "c = 30\n"
            ">>>>>>> REPLACE\n"
        )
        out = self.fn(code, patch)
        assert out is not None
        assert "a = 10" in out
        assert "b = 2" in out          # untouched middle line preserved
        assert "c = 30" in out

    def test_ambiguous_match_is_skipped(self):
        # 'x = 1' appears twice → should NOT be replaced (ambiguous)
        code = "x = 1\ny = 2\nx = 1\n"
        patch = (
            "<<<<<<< SEARCH\n"
            "x = 1\n"
            "=======\n"
            "x = 99\n"
            ">>>>>>> REPLACE\n"
        )
        out = self.fn(code, patch)
        # No unambiguous block applied → function returns None
        assert out is None

    def test_missing_search_text_returns_none(self):
        code = "print('hi')\n"
        patch = (
            "<<<<<<< SEARCH\n"
            "print('nonexistent')\n"
            "=======\n"
            "print('bye')\n"
            ">>>>>>> REPLACE\n"
        )
        assert self.fn(code, patch) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
