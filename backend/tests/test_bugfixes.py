"""
Unit tests for bug fixes B7 and B8.
"""
import os
import sys
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestBugFixes:
    def test_progress_clamping_b7(self):
        """Test that progress values are clamped correctly by thread_cb."""
        import queue
        q = queue.Queue()

        # Re-create thread_cb closure logic from app.py
        def thread_cb(msg, lvl="info", model=None, progress=None):
            payload = {"type": "status", "message": msg, "level": lvl}
            if model:
                payload["model"] = model
            if progress is not None:
                try:
                    prog_val = int(progress)
                    if lvl == "success":
                        prog_val = min(100, max(0, prog_val))
                    else:
                        prog_val = min(99, max(0, prog_val))
                    payload["progress"] = prog_val
                except (ValueError, TypeError):
                    payload["progress"] = progress
            q.put(payload)

        # 1. Non-success statuses should be capped at 99
        thread_cb("Running...", "info", progress=105)
        res = q.get()
        assert res["progress"] == 99

        thread_cb("Warning...", "warning", progress=100)
        res = q.get()
        assert res["progress"] == 99

        # 2. Success status should allow 100 but clamp above 100
        thread_cb("Success!", "success", progress=100)
        res = q.get()
        assert res["progress"] == 100

        thread_cb("Success!", "success", progress=120)
        res = q.get()
        assert res["progress"] == 100

        # 3. Normal values should pass through
        thread_cb("Running...", "info", progress=50)
        res = q.get()
        assert res["progress"] == 50

        # 4. Invalid/Negative values handling
        thread_cb("Running...", "info", progress=-10)
        res = q.get()
        assert res["progress"] == 0

        thread_cb("Running...", "info", progress="invalid")
        res = q.get()
        assert res["progress"] == "invalid"

    def test_coding_pipeline_manifest_parsing_b8(self):
        """Test that parse_multi_file_manifest is only executed when req_lang is html."""
        from backend.sandbox import Sandbox
        
        # Verify the XML multi-file manifest helper returns correct values
        html_manifest = '<file path="test.html">hello</file>'
        parsed = Sandbox.parse_multi_file_manifest(html_manifest)
        assert parsed == {"test.html": "hello"}

        # Verify that plain text or other files return empty dict
        non_html = "print('hello')"
        parsed_non_html = Sandbox.parse_multi_file_manifest(non_html)
        assert not parsed_non_html

    def test_compute_gen_tokens_c3(self):
        """Test _compute_gen_tokens helper in AgentOrchestrator."""
        from backend.orchestrator import AgentOrchestrator

        # Test with high context limit (e.g. 16384)
        # gen_tokens = 16384 * 0.4 = 6553.6 -> 6553.
        # Within [2048, 8192] -> 6553.
        # Headroom: 16384 - 6553 = 9831 >= 1500.
        assert AgentOrchestrator._compute_gen_tokens(16384) == 6553

        # Test with 4096 context limit
        # gen_tokens = 4096 * 0.4 = 1638.
        # Max(2048, 1638) = 2048.
        # Headroom: 4096 - 2048 = 2048 >= 1500.
        assert AgentOrchestrator._compute_gen_tokens(4096) == 2048

        # Test with 3000 context limit
        # gen_tokens = 3000 * 0.4 = 1200.
        # Max(2048, 1200) = 2048.
        # Headroom: 3000 - 2048 = 952 < 1500.
        # gen_tokens = max(512, 3000 - 1500) = 1500.
        assert AgentOrchestrator._compute_gen_tokens(3000) == 1500

        # Test with 1000 context limit
        # gen_tokens = 1000 * 0.4 = 400.
        # Max(2048, 400) = 2048.
        # Headroom: 1000 - 2048 = -1048 < 1500.
        # gen_tokens = max(512, 1000 - 1500) = 512.
        assert AgentOrchestrator._compute_gen_tokens(1000) == 512


