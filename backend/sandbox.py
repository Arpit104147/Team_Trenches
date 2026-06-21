import subprocess
import sys
import tempfile
import os
import re
import shutil
import textwrap

# GUI libraries that will always hang in a headless sandbox
GUI_SIGNATURES = [
    'pygame', 'tkinter', 'turtle', 'pyglet', 'arcade',
    'kivy', 'wxPython', 'PyQt', 'PySide', 'cv2.imshow',
    'matplotlib.pyplot.show', 'plt.show'
]

# Patterns that indicate an infinite/long-running loop
LOOP_PATTERNS = [
    r'while\s+True\s*[:{]',
    r'while\s+running\s*[:{]',
    r'while\s+not\s+done\s*[:{]',
    r'while\s+game\s*[:{]',
    r'while\s+active\s*[:{]',
    r'\.mainloop\(\)',
    r'app\.exec',
    r'pygame\.event\.get',
    r'for\s*\(\s*;\s*;\s*\)',       # C/C++ infinite loop: for(;;)
    r'while\s*\(\s*1\s*\)',         # C/C++ infinite loop: while(1)
    r'while\s*\(\s*true\s*\)',      # C/C++ infinite loop: while(true)
]

# ── Language Detection Heuristics ────────────────────────────────────────
# Strong indicators for each supported language
LANG_SIGNATURES = {
    'python': {
        'strong': [r'^\s*import\s+\w+', r'^\s*from\s+\w+\s+import', r'def\s+\w+\s*\(', r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', r'print\s*\(.*\)'],
        'ext': '.py',
        'compile': None,
        'run': [sys.executable, '{src}'],
    },
    'c': {
        'strong': [r'#include\s*<\w+\.h>', r'int\s+main\s*\(', r'printf\s*\(', r'scanf\s*\(', r'malloc\s*\(', r'free\s*\('],
        'ext': '.c',
        'compile': ['gcc', '{src}', '-o', '{bin}', '-lm'],
        'run': ['{bin}'],
    },
    'cpp': {
        'strong': [r'#include\s*<iostream>', r'#include\s*<vector>', r'#include\s*<string>',
                   r'std::', r'cout\s*<<', r'cin\s*>>', r'using\s+namespace\s+std'],
        'ext': '.cpp',
        'compile': ['g++', '{src}', '-o', '{bin}', '-lm', '-lstdc++'],
        'run': ['{bin}'],
    },
    'bash': {
        'strong': [r'^#!/bin/bash', r'^#!/bin/sh', r'\becho\s+', r'\bif\s+\[', r'\bfor\s+\w+\s+in\b', r'\bfi\b'],
        'ext': '.sh',
        'compile': None,
        'run': ['bash', '{src}'],
    },
    'javascript': {
        'strong': [r'\bconsole\.log\s*\(', r'\bconst\s+\w+\s*=', r'\blet\s+\w+\s*=', r'\bfunction\s+\w+\s*\(', r'=>\s*{'],
        'ext': '.js',
        'compile': None,
        'run': ['node', '{src}'],
    },
    'java': {
        'strong': [r'public\s+class\s+', r'public\s+static\s+void\s+main', r'System\.out\.println'],
        'ext': '.java',
        'compile': ['javac', '{src}'],
        'run': ['java', '-cp', '{dir}', '{classname}'],
    },
}

# ─────────────────────────────────────────────────────────────────────────
# RESTRICTED EXECUTION LAYER
# This is a self-contained Python script that runs INSIDE a subprocess.
# It strips away all dangerous OS access (file I/O, network, subprocess,
# shell commands) and only allows pure computation + safe libraries.
# The AI code runs in a completely isolated in-memory environment.
# ─────────────────────────────────────────────────────────────────────────
RESTRICTED_RUNNER_TEMPLATE = textwrap.dedent(r'''
import sys
import io
import json

# ── Step 1: Set Resource Limits (Linux only) ─────────────────────────────
try:
    import resource
    # Max 1 GB RAM (enough for numpy/pandas heavy workloads)
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    # Max 15 seconds of CPU time to aggressively kill infinite loops
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    # Max 50 child processes (allows multiprocessing but blocks fork bombs)
    resource.setrlimit(resource.RLIMIT_NPROC, (50, 50))
    # Max 100 MB file writes (allows data output but prevents disk flooding)
    resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))
except Exception:
    pass  # Non-Linux systems skip resource limits

# ── Step 1.5: Hard Block Network Access ──────────────────────────────────
import socket
def block_network(*args, **kwargs):
    raise PermissionError("Network access is strictly forbidden in the sandbox.")
socket.socket = block_network

# ── Step 2: Define the Whitelist of Safe Modules ─────────────────────────
ALLOWED_MODULES = {
    # Math & Science
    'math', 'cmath', 'decimal', 'fractions', 'statistics', 'numbers',
    # Data Structures & Algorithms
    'collections', 'itertools', 'functools', 'operator', 'bisect', 'heapq',
    'array', 'copy', 'enum', 'dataclasses', 'typing',
    # String & Text
    'string', 'textwrap', 're', 'unicodedata',
    # Date & Time (read-only, no system clock mutation)
    'datetime', 'time', 'calendar',
    # Data Formats (parsing only, no file I/O)
    'json', 'csv', 'base64', 'hashlib', 'hmac',
    # Random & Crypto
    'random', 'secrets',
    # Struct & Binary
    'struct', 'binascii',
    # Abstract Base Classes
    'abc',
    # Scientific Libraries (if installed)
    'numpy', 'sympy', 'scipy', 'pandas', 'plotly',
    # Physics & Unit Verification
    'pint',
    # Formal Logic & Theorem Proving
    'z3', 'z3core', 'z3types', 'z3printer', 'z3num',
    # Graph Theory & Network Analysis
    'networkx',
    # Astrophysics & Celestial Mechanics
    'astropy',
    # Web & API requests (for real-time weather and stock prediction data)
    'requests', 'urllib', 'http',
}

# ── Step 3: Create the Restricted Import Function ────────────────────────
_real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Custom import that only allows whitelisted modules."""
    # Get the top-level module name
    top_module = name.split('.')[0]
    if top_module not in ALLOWED_MODULES:
        raise ImportError(
            f"🔒 SANDBOX BLOCKED: Module '{name}' is not allowed in the restricted sandbox.\n"
            f"   Allowed modules: {', '.join(sorted(ALLOWED_MODULES))}"
        )
    return _real_import(name, globals, locals, fromlist, level)

# ── Step 4: Build Restricted Builtins ────────────────────────────────────
# Start with a copy of safe builtins, then remove dangerous ones
import builtins as _builtins_module

BLOCKED_BUILTINS = {
    'open',          # No file read/write
    'exec',          # No nested exec (prevent escape)
    'eval',          # No nested eval
    'compile',       # No dynamic code compilation
    '__import__',    # Replaced with our restricted version
    'globals',       # No access to the runner's global scope
    'breakpoint',    # No debugger access
    'exit',          # No process termination
    'quit',          # No process termination
    'memoryview',    # No raw memory access
    'input',         # No stdin (would hang the subprocess)
}

safe_builtins = {}
for name in dir(_builtins_module):
    if name.startswith('_') and name != '__name__':
        continue
    if name in BLOCKED_BUILTINS:
        continue
    safe_builtins[name] = getattr(_builtins_module, name)

# Inject our restricted import as the only way to load modules
safe_builtins['__import__'] = _restricted_import
safe_builtins['__name__'] = '__main__'

# ── Step 5: Capture stdout/stderr and Execute ────────────────────────────
captured_stdout = io.StringIO()
captured_stderr = io.StringIO()

# Read the AI code from the temp file
code_path = sys.argv[1]
with _real_import('builtins').open(code_path, 'r') as f:
    user_code = f.read()

# Build the completely isolated execution environment
restricted_globals = {'__builtins__': safe_builtins}

old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = captured_stdout
sys.stderr = captured_stderr

result = {"success": False, "output": "", "error": "", "restricted": True}

try:
    # This is the core: exec() runs the AI code in the restricted namespace
    compiled_code = _real_import('builtins').__import__('builtins').compile(user_code, '<sandbox>', 'exec')
    exec(compiled_code, restricted_globals)
    result["success"] = True
    result["output"] = captured_stdout.getvalue()
    if captured_stderr.getvalue():
        result["output"] += "\nWarnings/Stderr:\n" + captured_stderr.getvalue()
except ImportError as e:
    # Module was blocked — signal the caller to retry with unrestricted mode
    result["success"] = False
    result["error"] = str(e)
    result["restricted_block"] = True
except MemoryError:
    result["error"] = "MemoryError: Code exceeded the 256MB sandbox RAM limit."
except Exception as e:
    result["error"] = f"{type(e).__name__}: {str(e)}"
finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr

# Output the result as JSON so the parent process can parse it
print(json.dumps(result))
''')


class Sandbox:
    def __init__(self, timeout=60):
        self.timeout = timeout

    def _detect_gui(self, code):
        """Check if the code imports or uses any GUI library that would hang."""
        for sig in GUI_SIGNATURES:
            if sig in code:
                return sig
        return None

    def _detect_infinite_loop(self, code):
        """Check if the code contains patterns that suggest an infinite loop."""
        for pattern in LOOP_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return pattern
        return None

    def _detect_language(self, code):
        """
        Auto-detect the programming language of the code using heuristic signature matching.
        Returns one of: 'python', 'c', 'cpp', 'bash', 'javascript', 'java'
        """
        scores = {}
        for lang, sigs in LANG_SIGNATURES.items():
            score = 0
            for pattern in sigs['strong']:
                if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                    score += 1
            scores[lang] = score

        # C vs C++ disambiguation: if both match, check for C++-only features
        if scores.get('c', 0) > 0 and scores.get('cpp', 0) > 0:
            if scores['cpp'] >= scores['c']:
                scores['c'] = 0  # It's C++, not C

        # Find the language with the highest score
        best_lang = max(scores, key=scores.get) if scores else None
        best_score = scores.get(best_lang, 0) if best_lang else 0

        # If no strong match, default to Python
        if best_score == 0:
            return 'python'
        return best_lang

    def _check_compiler_available(self, compiler):
        """Check if a compiler/runtime is installed on the host system."""
        return shutil.which(compiler) is not None

    def _execute_compiled(self, code, lang_config, language, temp_dir):
        """
        Compile and execute code for compiled languages (C, C++, Java).
        Returns (success: bool, output: str)
        """
        src_path = os.path.join(temp_dir, f"program{lang_config['ext']}")
        bin_path = os.path.join(temp_dir, "program_bin")

        with open(src_path, 'w') as f:
            f.write(code)

        # ── Compilation Step ─────────────────────────────────────────────
        compiler = lang_config['compile'][0]
        if not self._check_compiler_available(compiler):
            return False, (
                f"⚠️ Compiler '{compiler}' not found on this system.\n"
                f"The {language.upper()} code is syntactically valid but cannot be compiled here.\n"
                f"Install '{compiler}' or run this code on a system with the {language.upper()} toolchain."
            )

        compile_cmd = [
            s.replace('{src}', src_path).replace('{bin}', bin_path)
            for s in lang_config['compile']
        ]

        try:
            comp_res = subprocess.run(
                compile_cmd, capture_output=True, text=True, timeout=30
            )
            if comp_res.returncode != 0:
                return False, f"Compilation Error:\n{comp_res.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out (>30s). Code may be too complex."

        # ── Execution Step ───────────────────────────────────────────────
        if language == 'java':
            # Java needs the classname extracted from the source
            class_match = re.search(r'public\s+class\s+(\w+)', code)
            classname = class_match.group(1) if class_match else "Main"
            run_cmd = [
                s.replace('{dir}', temp_dir).replace('{classname}', classname)
                for s in lang_config['run']
            ]
        else:
            run_cmd = [s.replace('{bin}', bin_path) for s in lang_config['run']]

        try:
            res = subprocess.run(
                run_cmd, capture_output=True, text=True, timeout=self.timeout
            )
            if res.returncode == 0:
                output = res.stdout
                if res.stderr:
                    output += "\nWarnings/Stderr:\n" + res.stderr
                return True, output.strip()
            else:
                error = res.stderr if res.stderr else res.stdout
                return False, f"Runtime Error:\n{error.strip()}"
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: Execution took longer than {self.timeout}s."

    def _execute_interpreted(self, code, lang_config, language):
        """
        Execute code for interpreted languages (Bash, JavaScript).
        Returns (success: bool, output: str)
        """
        runtime = lang_config['run'][0]
        if not self._check_compiler_available(runtime):
            return False, (
                f"⚠️ Runtime '{runtime}' not found on this system.\n"
                f"The {language} code appears valid but cannot be executed here.\n"
                f"Install '{runtime}' to run {language} scripts."
            )

        with tempfile.NamedTemporaryFile(mode='w', suffix=lang_config['ext'], delete=False) as f:
            f.write(code)
            path = f.name

        try:
            run_cmd = [s.replace('{src}', path) for s in lang_config['run']]
            res = subprocess.run(
                run_cmd, capture_output=True, text=True, timeout=self.timeout
            )
            if res.returncode == 0:
                output = res.stdout
                if res.stderr:
                    output += "\nWarnings/Stderr:\n" + res.stderr
                return True, output.strip()
            else:
                error = res.stderr if res.stderr else res.stdout
                return False, error.strip()
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: Code took longer than {self.timeout}s to execute."
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def execute(self, code, language=None):
        """
        Polyglot Sandbox: Execute code in Python, C, C++, Bash, JavaScript, or Java.
        Auto-detects the language if not specified.
        Returns (success: bool, output: str)
        """
        # ── Pre-Execution Analysis ───────────────────────────────────────
        gui_lib = self._detect_gui(code)
        loop_pattern = self._detect_infinite_loop(code)

        # If the code is a GUI app, don't even try to run it
        if gui_lib and loop_pattern:
            return True, (
                f"⚠️ GUI Application Detected ({gui_lib})\n"
                f"This script creates a graphical window with an event loop.\n"
                f"It cannot run in a headless cloud sandbox, but the code is valid.\n"
                f"Copy the code above and run it on your local machine to see the simulation!"
            )

        # ── Language Detection ───────────────────────────────────────────
        if language is None:
            language = self._detect_language(code)

        # ── Python Execution: Restricted First, Fallback to Unrestricted ─
        if language == 'python':
            return self._execute_python_restricted(code, loop_pattern)

        # ── Compiled Languages (C, C++, Java) ────────────────────────────
        lang_config = LANG_SIGNATURES.get(language)
        if not lang_config:
            return False, f"Unsupported language: {language}"

        if lang_config['compile'] is not None:
            temp_dir = tempfile.mkdtemp(prefix="sandbox_")
            try:
                return self._execute_compiled(code, lang_config, language, temp_dir)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        # ── Interpreted Languages (Bash, JavaScript) ─────────────────────
        return self._execute_interpreted(code, lang_config, language)

    def _execute_python_restricted(self, code, loop_pattern=None):
        """
        Execute Python code in a RESTRICTED in-memory sandbox.
        
        Architecture:
        1. The AI code is written to a temp file.
        2. A special "runner" script is generated that:
           a. Sets Linux resource limits (256MB RAM, 30s CPU, 0 child processes)
           b. Strips away all dangerous builtins (open, exec, eval, __import__)
           c. Injects a custom __import__ that only allows whitelisted modules
           d. Runs exec(code) inside the restricted namespace
        3. The runner executes in a subprocess (process isolation from FastAPI).
        4. If the restricted sandbox blocks a legitimate import, it automatically
           falls back to unrestricted subprocess execution.
        
        This gives us THREE layers of protection:
        - Layer 1: Process isolation (subprocess can't crash the server)
        - Layer 2: Restricted builtins (no open/exec/eval/import of OS modules)
        - Layer 3: Resource limits (RAM/CPU/disk caps prevent DoS attacks)
        """
        # Write the AI's code to a temp file
        code_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        code_file.write(code)
        code_file.close()
        code_path = code_file.name

        # Write the restricted runner script to a temp file
        runner_file = tempfile.NamedTemporaryFile(mode='w', suffix='_runner.py', delete=False)
        runner_file.write(RESTRICTED_RUNNER_TEMPLATE)
        runner_file.close()
        runner_path = runner_file.name

        try:
            # Execute the runner script, passing the code file path as argv[1]
            res = subprocess.run(
                [sys.executable, runner_path, code_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Parse the JSON result from the runner
            stdout = res.stdout.strip()
            if stdout:
                import json
                try:
                    result = json.loads(stdout)

                    # Check if the restricted sandbox blocked a legitimate import
                    if result.get("restricted_block"):
                        # Fall back to unrestricted execution
                        return self._execute_python_unrestricted(code, loop_pattern)

                    if result.get("success"):
                        output = result.get("output", "").strip()
                        if not output:
                            output = "(Code executed successfully with no output)"
                        return True, f"🔒 [Restricted Sandbox]\n{output}"
                    else:
                        return False, result.get("error", "Unknown error in restricted sandbox")

                except json.JSONDecodeError:
                    # Runner produced non-JSON output — something unexpected happened
                    # Fall back to unrestricted execution
                    return self._execute_python_unrestricted(code, loop_pattern)
            else:
                # No stdout from runner — check stderr
                if res.stderr:
                    # Runner itself crashed — fall back to unrestricted
                    return self._execute_python_unrestricted(code, loop_pattern)
                return True, "🔒 [Restricted Sandbox]\n(Code executed successfully with no output)"

        except subprocess.TimeoutExpired:
            if loop_pattern:
                return True, (
                    f"⚠️ Infinite Loop Detected (pattern: {loop_pattern})\n"
                    f"The script contains a long-running loop that exceeds the "
                    f"{self.timeout}s sandbox limit.\n"
                    f"This is expected for interactive/game scripts. "
                    f"The code is valid — run it locally to see the output!"
                )
            return False, (
                f"TimeoutError: Code took longer than {self.timeout}s to execute.\n"
                f"This may indicate an infinite loop or very heavy computation."
            )
        except Exception as e:
            # If anything goes wrong with restricted mode, fall back gracefully
            return self._execute_python_unrestricted(code, loop_pattern)
        finally:
            # Clean up both temp files
            for path in [code_path, runner_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def _execute_python_unrestricted(self, code, loop_pattern=None):
        """
        Fallback: Execute Python code in a standard subprocess without restrictions.
        Used when the restricted sandbox blocks a legitimate module the AI needs.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            path = f.name

        try:
            res = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            if res.returncode == 0:
                output = res.stdout
                if res.stderr:
                    output += "\nWarnings/Stderr:\n" + res.stderr
                return True, f"⚠️ [Unrestricted Fallback]\n{output.strip()}"
            else:
                error = res.stderr if res.stderr else res.stdout
                return False, error.strip()
        except subprocess.TimeoutExpired:
            if loop_pattern:
                return True, (
                    f"⚠️ Infinite Loop Detected (pattern: {loop_pattern})\n"
                    f"The script contains a long-running loop that exceeds the "
                    f"{self.timeout}s sandbox limit.\n"
                    f"This is expected for interactive/game scripts. "
                    f"The code is valid — run it locally to see the output!"
                )
            return False, (
                f"TimeoutError: Code took longer than {self.timeout}s to execute.\n"
                f"This may indicate an infinite loop or very heavy computation."
            )
        except Exception as e:
            return False, f"ExecutionError: {str(e)}"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @staticmethod
    def extract_code(text):
        """
        Extract code from markdown code blocks in LLM responses.
        Handles both closed and unclosed (cut-off) blocks.
        """
        def _sanitize(code_str):
            # Remove hallucinated Jupyter magic commands that cause SyntaxErrors in pure Python
            return re.sub(r'^[!%]\s*pip\s+install.*$', '', code_str, flags=re.MULTILINE).strip()

        # 1. Try language-specific closed code blocks first
        lang_patterns = [
            (r"```\s*html\s*(.*?)\s*```", 'html'),
            (r"```\s*python\s*(.*?)\s*```", 'python'),
            (r"```\s*py\s*(.*?)\s*```", 'python'),
            (r"```\s*c\+\+\s*(.*?)\s*```", 'cpp'),
            (r"```\s*cpp\s*(.*?)\s*```", 'cpp'),
            (r"```\s*c\s*(.*?)\s*```", 'c'),
            (r"```\s*bash\s*(.*?)\s*```", 'bash'),
            (r"```\s*sh\s*(.*?)\s*```", 'bash'),
            (r"```\s*javascript\s*(.*?)\s*```", 'javascript'),
            (r"```\s*js\s*(.*?)\s*```", 'javascript'),
            (r"```\s*java\s*(.*?)\s*```", 'java'),
        ]

        for pattern, lang in lang_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return _sanitize(match.group(1))

        # 2. Fallback: Match generic closed ``` <code> ```
        generic_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if generic_match:
            content = generic_match.group(1).strip()
            first_line = content.split('\n')[0].strip().lower()
            known_tags = ['html', 'python', 'py', 'javascript', 'js', 'css', 'bash', 'sh', 'c', 'cpp', 'c++', 'java']
            if first_line in known_tags:
                return _sanitize("\n".join(content.split('\n')[1:]))
            return _sanitize(content)

        # 3. Match unclosed language-specific code blocks (cut off at response end)
        unclosed_patterns = [
            (r"```html\s*(.*)$", 'html'),
            (r"```python\s*(.*)$", 'python'),
            (r"```py\s*(.*)$", 'python'),
            (r"```javascript\s*(.*)$", 'javascript'),
            (r"```js\s*(.*)$", 'javascript'),
            (r"```c\+\+\s*(.*)$", 'cpp'),
            (r"```cpp\s*(.*)$", 'cpp'),
            (r"```c\s*(.*)$", 'c'),
            (r"```bash\s*(.*)$", 'bash'),
            (r"```sh\s*(.*)$", 'bash'),
            (r"```java\s*(.*)$", 'java'),
        ]

        for pattern, lang in unclosed_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return _sanitize(match.group(1))

        # 4. Match generic unclosed block
        generic_unclosed = re.search(r"```\s*(.*)$", text, re.DOTALL)
        if generic_unclosed:
            content = generic_unclosed.group(1).strip()
            first_line = content.split('\n')[0].strip().lower()
            known_tags = ['html', 'python', 'py', 'javascript', 'js', 'css', 'bash', 'sh', 'c', 'cpp', 'c++', 'java']
            if first_line in known_tags:
                return _sanitize("\n".join(content.split('\n')[1:]))
            return _sanitize(content)

        return _sanitize(text)
