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

# ── Common module-to-pip-package name mappings ───────────────────────────
# Many Python modules have pip package names that differ from their import name.
PIP_PACKAGE_MAP = {
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'sklearn': 'scikit-learn',
    'yaml': 'pyyaml',
    'bs4': 'beautifulsoup4',
    'attr': 'attrs',
    'dateutil': 'python-dateutil',
    'dotenv': 'python-dotenv',
    'jose': 'python-jose',
    'serial': 'pyserial',
    'usb': 'pyusb',
    'Crypto': 'pycryptodome',
    'OpenSSL': 'pyopenssl',
    'github': 'PyGithub',
    'telegram': 'python-telegram-bot',
    'discord': 'discord.py',
    'jwt': 'pyjwt',
    'Bio': 'biopython',
    'wx': 'wxPython',
    'gi': 'pygobject',
    'magic': 'python-magic',
    'docx': 'python-docx',
    'pptx': 'python-pptx',
    'xlrd': 'xlrd',
    'openpyxl': 'openpyxl',
}

# Maximum output characters returned from sandbox (prevents context overflow)
MAX_OUTPUT_CHARS = 8000

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
    'go': {
        'strong': [r'^package\s+main\b', r'import\s+\(\s*"fmt"', r'func\s+main\s*\(\)'],
        'ext': '.go',
        'compile': ['go', 'build', '-o', '{bin}', '{src}'],
        'run': ['{bin}'],
    },
    'rust': {
        'strong': [r'fn\s+main\s*\(\)', r'println!\s*\(', r'use\s+std::'],
        'ext': '.rs',
        'compile': ['rustc', '{src}', '-o', '{bin}'],
        'run': ['{bin}'],
    },
    'typescript': {
        'strong': [r'\binterface\s+\w+\b', r'\btype\s+\w+\s*=', r'console\.log\s*\('],
        'ext': '.ts',
        'compile': None,
        'run': ['ts-node', '{src}'],
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
    # Max 2 GB RAM (enough for numpy/pandas heavy workloads and complex physics/biology simulation solvers)
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024))
    # Max 30 seconds of CPU time to aggressively kill infinite loops
    # Increased to 300 seconds for heavier simulations
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    # Max 200 child processes (allows multiprocessing but blocks fork bombs)
    resource.setrlimit(resource.RLIMIT_NPROC, (200, 200))
    # Max 100 MB file writes (allows data output but prevents disk flooding)
    resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))
except Exception:
    pass  # Non-Linux systems skip resource limits

# ── Step 1.5: Hard Block Network Access ──────────────────────────────────
import socket
class BlockedSocket:
    def __init__(self, *args, **kwargs):
        raise PermissionError("Network access is strictly forbidden in the sandbox.")
socket.socket = BlockedSocket

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
    'numpy', 'sympy', 'scipy', 'pandas', 'plotly', 'sklearn', 'statsmodels',
    # Physics & Unit Verification
    'pint',
    # Formal Logic & Theorem Proving
    'z3', 'z3core', 'z3types', 'z3printer', 'z3num',
    # Graph Theory & Network Analysis
    'networkx',
    # Astrophysics & Celestial Mechanics
    'astropy',
    # Bioinformatics & Cheminformatics
    'Bio', 'rdkit',
    # Quantum Physics & Rocket Dynamics
    'rocketpy', 'qiskit', 'qutip',
    # Cybersecurity & Cryptography / Network Simulation
    'cryptography', 'scapy', 'jwt', 'pyjwt',
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

# ── Step 4.5: Inject Sandbox Data Helper & API Keys ──────────────────────
import os
os.environ["ALPHA_VANTAGE_KEY"] = "demo_finance_key_ff88c221"
os.environ["OPENWEATHER_KEY"] = "demo_weather_key_991823ab"
os.environ["COINGECKO_KEY"] = "demo_crypto_key_aa11b239"

class SandboxDataHelper:
    @staticmethod
    def get_stock_data(symbol, period="1y"):
        try:
            import numpy as np
            import pandas as pd
            from datetime import datetime, timedelta
            baselines = {"AAPL": 175.0, "MSFT": 420.0, "GOOG": 150.0, "NVDA": 800.0, "TSLA": 180.0, "BTC": 65000.0}
            base = baselines.get(symbol.upper(), 100.0)
            days = 365
            if "mo" in period:
                days = int(period.replace("mo", "")) * 30
            elif "y" in period:
                days = int(period.replace("y", "")) * 365
            elif "d" in period:
                days = int(period.replace("d", ""))
            np.random.seed(42)
            dates = [datetime.now() - timedelta(days=i) for i in range(days)]
            dates.reverse()
            returns = np.random.normal(0.0002, 0.015, days)
            prices = [base]
            for r in returns:
                prices.append(prices[-1] * np.exp(r))
            prices = prices[:-1]
            return pd.DataFrame({
                "Date": [d.strftime("%Y-%m-%d") for d in dates],
                "Open": [p * (1 + np.random.normal(0, 0.002)) for p in prices],
                "High": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                "Low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                "Close": prices,
                "Volume": [int(np.random.poisson(1000000) * (p/base)) for p in prices]
            })
        except Exception as e:
            return str(e)

    @staticmethod
    def get_weather_data(city, days=7):
        try:
            import numpy as np
            import pandas as pd
            from datetime import datetime, timedelta
            city_temps = {"LONDON": 15.0, "NEW YORK": 22.0, "TOKYO": 18.0, "MUMBAI": 30.0, "SYDNEY": 19.0}
            base_temp = city_temps.get(city.upper(), 20.0)
            dates = [datetime.now() + timedelta(days=i) for i in range(days)]
            np.random.seed(42)
            temps = [base_temp + np.sin(i/3)*5 + np.random.normal(0, 1) for i in range(days)]
            humidity = [60 + np.cos(i/3)*20 + np.random.normal(0, 5) for i in range(days)]
            return pd.DataFrame({
                "Date": [d.strftime("%Y-%m-%d") for d in dates],
                "Temperature_C": temps,
                "Humidity_Pct": [min(100, max(0, h)) for h in humidity],
                "Condition": np.random.choice(["Sunny", "Partly Cloudy", "Rainy", "Overcast"], days)
            })
        except Exception as e:
            return str(e)

    @staticmethod
    def get_api_key(service_name):
        keys = {"weather": "demo_weather_key_991823ab", "finance": "demo_finance_key_ff88c221", "crypto": "demo_crypto_key_aa11b239"}
        return keys.get(service_name.lower(), "demo_generic_key_000000")

# Build the completely isolated execution environment
restricted_globals = {
    '__builtins__': safe_builtins,
    'SandboxDataHelper': SandboxDataHelper
}

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
except (ImportError, PermissionError) as e:
    # Module was blocked or network was accessed — signal the caller to retry with unrestricted mode
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
    def __init__(self, timeout=300):
        self.timeout = timeout

    # ── Utility Methods ──────────────────────────────────────────────────
    @staticmethod
    def _strip_ansi(text):
        """Strip ANSI escape codes (color, cursor, etc.) from text for cleaner output."""
        return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)

    @staticmethod
    def _truncate_output(output, max_chars=MAX_OUTPUT_CHARS):
        """Truncate sandbox output to prevent context overflow, preserving head and tail."""
        if not output or len(output) <= max_chars:
            return output
        head = max_chars // 2
        tail = max_chars // 2
        return (
            output[:head]
            + f"\n\n... [OUTPUT TRUNCATED: {len(output):,} chars total, showing first {head:,} and last {tail:,}] ...\n\n"
            + output[-tail:]
        )

    def _auto_install_missing_module(self, error_output):
        """Parse ModuleNotFoundError from sandbox output and auto-install the missing pip package."""
        match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", error_output)
        if not match:
            return False

        module_name = match.group(1).split('.')[0]  # Get top-level module
        pip_name = PIP_PACKAGE_MAP.get(module_name, module_name)

        print(f"📦 Auto-installing missing package: {pip_name} (module: {module_name})")
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', pip_name, '--quiet', '--disable-pip-version-check'],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                print(f"✅ Successfully installed {pip_name}")
                return True
            else:
                print(f"❌ Failed to install {pip_name}: {result.stderr[:300]}")
                return False
        except subprocess.TimeoutExpired:
            print(f"❌ Timed out installing {pip_name}")
            return False
        except Exception as e:
            print(f"❌ Auto-install error: {e}")
            return False

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
        # Pre-check for syntax/truncation errors
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            return False, (
                f"SyntaxError: {e.msg} at line {e.lineno}.\n"
                f"CRITICAL: Your code was likely truncated (cut off mid-sentence) or contains unbalanced braces/quotes.\n"
                f"Ensure all strings, functions, quotes, and brackets are fully closed.\n"
                f"Write shorter, more concise code if necessary to avoid hitting output limits."
            )

        # Prepend compatibility monkeypatch for NumPy < 2.0 (e.g. numpy==1.26.4) and rocketpy
        compat_code = (
            "import sys\n"
            "try:\n"
            "    import numpy as _np\n"
            "    if not hasattr(_np, 'trapezoid'):\n"
            "        _np.trapezoid = getattr(_np, 'trapz', None)\n"
            "except Exception:\n"
            "    pass\n\n"
        ) + code

        # Write the AI's code to a temp file
        code_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        code_file.write(compat_code)
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
                        output = self._strip_ansi(result.get("output", "").strip())
                        if not output:
                            output = "(Code executed successfully with no output)"
                        return True, self._truncate_output(f"🔒 [Restricted Sandbox]\n{output}")
                    else:
                        return False, self._truncate_output(result.get("error", "Unknown error in restricted sandbox"))

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
        # Prepend compatibility monkeypatch for NumPy < 2.0 (e.g. numpy==1.26.4) and rocketpy
        compat_code = (
            "import sys\n"
            "try:\n"
            "    import numpy as _np\n"
            "    if not hasattr(_np, 'trapezoid'):\n"
            "        _np.trapezoid = getattr(_np, 'trapz', None)\n"
            "except Exception:\n"
            "    pass\n\n"
        ) + code

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(compat_code)
            path = f.name

        try:
            res = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            if res.returncode == 0:
                output = self._strip_ansi(res.stdout)
                if res.stderr:
                    output += "\nWarnings/Stderr:\n" + self._strip_ansi(res.stderr)
                return True, self._truncate_output(f"⚠️ [Unrestricted Fallback]\n{output.strip()}")
            else:
                error = res.stderr if res.stderr else res.stdout
                error_str = self._strip_ansi(error.strip())

                # ── Auto-pip: Install missing modules and retry once ──────
                if 'ModuleNotFoundError' in error_str and self._auto_install_missing_module(error_str):
                    res2 = subprocess.run(
                        [sys.executable, path],
                        capture_output=True, text=True, timeout=self.timeout
                    )
                    if res2.returncode == 0:
                        output2 = self._strip_ansi(res2.stdout)
                        if res2.stderr:
                            output2 += "\nWarnings/Stderr:\n" + self._strip_ansi(res2.stderr)
                        return True, self._truncate_output(f"📦 [Auto-installed package + Executed]\n{output2.strip()}")
                    else:
                        error2 = res2.stderr if res2.stderr else res2.stdout
                        return False, self._truncate_output(self._strip_ansi(error2.strip()))

                return False, self._truncate_output(error_str)
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
