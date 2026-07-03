"""
Dependency-free test runner for backend/tests/.

Runs any Test* class's test_* methods without requiring pytest. Useful when
the developer environment doesn't have pytest installed. Exits non-zero on
any failure so it's still CI-friendly.

Usage:
    python backend/tests/run_tests.py
"""
import importlib
import inspect
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class _SkipModule(Exception):
    """Raised when a test module opts out (e.g. missing heavy deps)."""


def _install_pytest_stub():
    """Provide a minimal `pytest` stub so test files that `import pytest`
    for `pytest.skip(..., allow_module_level=True)` still load."""
    if "pytest" in sys.modules:
        return

    import types
    import importlib.machinery

    stub = types.ModuleType("pytest")
    stub.__spec__ = importlib.machinery.ModuleSpec("pytest", None)

    def skip(msg="", allow_module_level=False):
        raise _SkipModule(msg)

    def fixture(*a, **kw):  # no-op decorator
        def deco(fn):
            return fn
        if a and callable(a[0]):
            return a[0]
        return deco

    def raises(*a, **kw):
        # Not needed by our current tests; provide a stub that fails loudly.
        raise RuntimeError("pytest.raises stub — not implemented in run_tests.py")

    stub.skip = skip
    stub.fixture = fixture
    stub.raises = raises
    stub.main = lambda *a, **kw: 0  # so `if __name__ == '__main__'` blocks are no-ops
    sys.modules["pytest"] = stub


def _discover_test_modules():
    files = []
    for name in sorted(os.listdir(_HERE)):
        if name.startswith("test_") and name.endswith(".py"):
            files.append(name[:-3])
    return files


def _run_module(mod_name):
    """Run all Test*.test_* methods in one module. Returns (passed, failed, skipped)."""
    passed = failed = 0
    try:
        module = importlib.import_module(f"backend.tests.{mod_name}")
    except _SkipModule as e:
        print(f"  ↷ SKIP module {mod_name}: {e}")
        return 0, 0, 1
    except Exception:
        print(f"  ✗ IMPORT FAILED {mod_name}")
        traceback.print_exc()
        return 0, 1, 0

    for cls_name, cls in inspect.getmembers(module, inspect.isclass):
        if not cls_name.startswith("Test"):
            continue
        for method_name, method in inspect.getmembers(cls, inspect.isfunction):
            if not method_name.startswith("test_"):
                continue
            instance = cls()
            if hasattr(instance, "setup_method"):
                try:
                    instance.setup_method()
                except Exception:
                    print(f"  ✗ {cls_name}.setup_method")
                    traceback.print_exc()
                    failed += 1
                    continue
            label = f"{mod_name}.{cls_name}.{method_name}"
            try:
                method(instance)
                print(f"  ✓ {label}")
                passed += 1
            except Exception:
                print(f"  ✗ {label}")
                traceback.print_exc()
                failed += 1
    return passed, failed, 0


def main():
    _install_pytest_stub()
    total_p = total_f = total_s = 0
    modules = _discover_test_modules()
    print(f"Discovered {len(modules)} test module(s): {modules}\n")
    for m in modules:
        print(f"── {m} ──")
        p, f, s = _run_module(m)
        total_p += p
        total_f += f
        total_s += s
        print()
    print("=" * 60)
    print(f"RESULT: {total_p} passed, {total_f} failed, {total_s} skipped")
    print("=" * 60)
    return 0 if total_f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
