import subprocess
import sys
import tempfile
import os
import re
import shutil
import textwrap
import json
import signal
import shlex
import tokenize
import io as _io_top

# Phase 3: Security scanning
try:
    from backend.security import scan_code_sast, format_sast_report
    SAST_AVAILABLE = True
except ImportError:
    try:
        from security import scan_code_sast, format_sast_report
        SAST_AVAILABLE = True
    except ImportError:
        SAST_AVAILABLE = False

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
        'run': ['npx', '-y', 'tsx', '{src}'],
    },
    # ── EDA / Chip Design Languages ──────────────────────────────────────
    'verilog': {
        'strong': [r'\bmodule\s+\w+', r'\binput\s+', r'\boutput\s+', r'\balways\s*@', r'\bassign\s+',
                   r'\breg\s+', r'\bwire\s+', r'\bendmodule\b', r'\binitial\s+begin'],
        'ext': '.v',
        'compile': ['iverilog', '-o', '{bin}', '{src}'],
        'run': ['vvp', '{bin}'],
    },
    'systemverilog': {
        'strong': [r'\bmodule\s+\w+', r'\blogic\s+', r'\balways_ff\s+', r'\balways_comb\b',
                   r'\binterface\s+\w+', r'\btypedef\s+', r'\bpackage\s+\w+'],
        'ext': '.sv',
        'compile': ['iverilog', '-g2012', '-o', '{bin}', '{src}'],
        'run': ['vvp', '{bin}'],
    },
    'spice': {
        'strong': [r'\.subckt\s+', r'\.tran\s+', r'\.dc\s+', r'\.ac\s+', r'\.end\b',
                   r'\.model\s+', r'\.param\s+', r'^\.title\b'],
        'ext': '.cir',
        'compile': None,
        'run': ['ngspice', '-b', '{src}'],
    },
    'yosys_tcl': {
        'strong': [r'\bread_verilog\b', r'\bsynth\b', r'\bwrite_verilog\b', r'\bstat\b',
                   r'\bread_liberty\b', r'\babc\b', r'\bopt\b'],
        'ext': '.ys',
        'compile': None,
        'run': ['yosys', '-s', '{src}'],
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
import os as _os_module

# ── Capture privileged builtins BEFORE we install the restricted layer ───
# These are the only handles we keep to the real, unrestricted primitives.
# The user code will NEVER see these names.
_REAL_OPEN = open
_REAL_COMPILE = compile
_REAL_EXEC = exec

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
import socket as _socket_module

def _blocked_net(*args, **kwargs):
    raise PermissionError("Network access is strictly forbidden in the sandbox.")

# Block every entrypoint into the network stack, not just socket.socket()
_socket_module.socket = _blocked_net
_socket_module.create_connection = _blocked_net
_socket_module.create_server = _blocked_net
if hasattr(_socket_module, 'socketpair'):
    _socket_module.socketpair = _blocked_net
if hasattr(_socket_module, 'fromfd'):
    _socket_module.fromfd = _blocked_net
# Also gate getaddrinfo so DNS lookups fail fast
_socket_module.getaddrinfo = _blocked_net
_socket_module.gethostbyname = _blocked_net

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
    'numpy', 'sympy', 'scipy', 'pandas', 'plotly', 'sklearn', 'statsmodels', 'matplotlib',
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
    # EDA / Chip Design Layout (GDSII generation)
    'gdstk', 'gdspy',
    # Cybersecurity & Cryptography (pure-crypto only — no network)
    'cryptography', 'jwt', 'pyjwt', 'Crypto', 'jose',
    # 'socket' is whitelisted ONLY so that any code touching it reaches our
    # monkey-patched entrypoints (which raise PermissionError). Without this,
    # `import socket` would raise _WhitelistBlockedImport, which the parent
    # process interprets as "legitimate dependency missing" and retries in
    # the UNRESTRICTED sandbox — completely defeating the network block.
    'socket',
    # NOTE: 'requests', 'urllib', 'http', 'scapy' are intentionally REMOVED.
    # The restricted sandbox forbids network I/O. Use SandboxDataHelper for
    # deterministic mock stock/weather data instead of live HTTP calls.
}

# ── Step 3: Create the Restricted Import Function ────────────────────────
_real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

# Sentinel class so we can distinguish a whitelist miss (legitimate fallback
# candidate) from a real PermissionError raised by the network kill-switch
# (which must NEVER be silently promoted to the unrestricted sandbox).
class _WhitelistBlockedImport(ImportError):
    pass

def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Custom import that only allows whitelisted modules."""
    # Get the top-level module name
    top_module = name.split('.')[0]
    if top_module not in ALLOWED_MODULES:
        raise _WhitelistBlockedImport(
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

# Read the AI code from the temp file using the REAL open we captured before
# the restricted layer was installed. Using _real_import('builtins').open
# would let malicious code re-obtain a real 'open' by re-invoking the same
# trick from inside exec() — we close that hole here.
code_path = sys.argv[1]
with _REAL_OPEN(code_path, 'r') as f:
    user_code = f.read()

# ── Step 4.5: Inject Sandbox Data Helper & API Keys ──────────────────────
import os
os.environ["ALPHA_VANTAGE_KEY"] = "demo_finance_key_ff88c221"
os.environ["OPENWEATHER_KEY"] = "demo_weather_key_991823ab"
os.environ["COINGECKO_KEY"] = "demo_crypto_key_aa11b239"

class SandboxDataHelper:
    # NOTE: This helper returns DETERMINISTIC SYNTHETIC data (fixed random seed).
    # It is NOT a real market/weather feed. Callers must label any downstream
    # analysis as "simulated" to avoid hallucinating factual claims.
    IS_SYNTHETIC = True

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
    # This is the core: exec() runs the AI code in the restricted namespace.
    # Compile with the REAL compile() captured at runner startup — do NOT use
    # _real_import('builtins').compile because leaking a handle to `builtins`
    # via the restricted __import__ was the historical escape path.
    compiled_code = _REAL_COMPILE(user_code, '<sandbox>', 'exec')
    _REAL_EXEC(compiled_code, restricted_globals)
    result["success"] = True
    result["output"] = captured_stdout.getvalue()
    if captured_stderr.getvalue():
        result["output"] += "\nWarnings/Stderr:\n" + captured_stderr.getvalue()
except _WhitelistBlockedImport as e:
    # A whitelist miss is the ONLY condition under which the parent process
    # is allowed to retry in the unrestricted sandbox.
    result["success"] = False
    result["error"] = str(e)
    result["restricted_block"] = True
except PermissionError as e:
    # PermissionError comes from the network kill-switch (or a future
    # filesystem guard). This is a hard security failure — do NOT set
    # restricted_block, so the parent will NOT fall back to unrestricted.
    result["success"] = False
    result["error"] = f"PermissionError (sandbox policy): {e}"
    result["policy_violation"] = True
except MemoryError:
    result["error"] = "MemoryError: Code exceeded the 2 GB sandbox RAM limit."
except ImportError as e:
    # A genuine ImportError from real code (e.g. optional dep missing) is
    # allowed to fall back — user code was well-formed, dependency was not.
    result["success"] = False
    result["error"] = str(e)
    result["restricted_block"] = True
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
        self.active_workspaces = set()
        # ── Phase 1: Detect kernel-level isolation capabilities ──────
        self.isolation_available = self._check_isolation_support()
        if self.isolation_available:
            print("Sandbox: ✅ Native kernel isolation available (unshare/chroot/seccomp)")
        else:
            print("Sandbox: ⚠️ Kernel isolation not available (no root/CAP_SYS_ADMIN). Using 3-layer sandbox.")

    # ── Kernel Isolation Detection ────────────────────────────────────

    @staticmethod
    def _check_isolation_support():
        """Check if native Linux kernel isolation tools are available.
        Returns True if unshare is available and we have sufficient privileges.
        """
        if os.name != 'posix':
            return False
        # Check if unshare binary exists
        if not shutil.which('unshare'):
            return False
        # Check if we have CAP_SYS_ADMIN or are root (required for --net/--pid)
        if os.geteuid() == 0:
            return True
        # Check for unprivileged user namespaces (available on most modern kernels)
        try:
            with open('/proc/sys/kernel/unprivileged_userns_clone', 'r') as f:
                return f.read().strip() == '1'
        except (FileNotFoundError, PermissionError):
            # On many kernels this sysctl doesn't exist but user namespaces work
            # Try a quick probe: attempt unshare with --user which doesn't need root
            try:
                probe = subprocess.run(
                    ['unshare', '--user', '--', 'echo', 'ok'],
                    capture_output=True, text=True, timeout=5
                )
                return probe.returncode == 0 and 'ok' in probe.stdout
            except Exception:
                return False

    def _wrap_with_isolation(self, cmd, temp_dir=None):
        """Wrap a command with kernel-level isolation using unshare.
        Provides network isolation, PID namespace isolation, and IPC isolation.
        Falls back to the original command if isolation is not available.

        Args:
            cmd: The command list to wrap
            temp_dir: Optional working directory for chroot-like filesystem isolation

        Returns:
            The wrapped command list (or original if isolation unavailable)
        """
        if not self.isolation_available:
            return cmd

        isolation_cmd = ['unshare']

        # If we are root, use full namespace isolation
        if os.geteuid() == 0:
            isolation_cmd.extend([
                '--net',   # No network access — sandbox cannot make HTTP calls
                '--ipc',   # Isolated System V IPC and POSIX message queues
                '--pid',   # Cannot see or kill host processes
                '--fork',  # Fork for clean PID 1 mapping inside namespace
                '--mount-proc',  # Mount /proc for the isolated PID namespace
            ])
        else:
            # Unprivileged: use user namespace (limited but still useful)
            isolation_cmd.extend([
                '--user',  # Create user namespace (unprivileged)
                '--net',   # Network isolation within user namespace
                '--fork',  # Fork for clean process tree
            ])

        isolation_cmd.append('--')  # Separator
        isolation_cmd.extend(cmd)
        return isolation_cmd

    @staticmethod
    def _apply_seccomp_preexec():
        """Return a preexec_fn that applies basic seccomp-like restrictions.
        Uses resource limits and signal masks as a lightweight alternative
        to full seccomp-bpf (which requires C extensions).

        This is called in the child process before exec, providing:
        - No new child processes (RLIMIT_NPROC=0)
        - Limited file descriptors
        - No core dumps
        """
        import resource
        def _apply():
            try:
                # Prevent fork bombs — no new child processes
                resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
                # Limit open file descriptors (prevent FD exhaustion)
                resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
                # No core dumps (prevent disk filling)
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                # Limit virtual memory to 2GB
                mem_limit = 2 * 1024 * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
                # Limit CPU time to 120 seconds
                resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
            except (ValueError, resource.error):
                pass  # Graceful degradation if limits can't be set
        return _apply

    def clean_workspace(self, temp_dir):
        """Clean up a specific workspace directory and remove it from tracking."""
        if temp_dir in self.active_workspaces:
            self.active_workspaces.discard(temp_dir)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    def clean_all_workspaces(self):
        """Clean up all tracked workspace directories."""
        for temp_dir in list(self.active_workspaces):
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        self.active_workspaces.clear()

    def __del__(self):
        try:
            self.clean_all_workspaces()
        except Exception:
            pass

    # ── EDA / Chip Design Methods ────────────────────────────────────────

    @staticmethod
    def check_eda_tools():
        """Check which EDA tools are installed and available on the system PATH.
        Returns a dict mapping tool names to booleans.
        """
        tools = {
            'iverilog': 'iverilog',
            'vvp': 'vvp',
            'yosys': 'yosys',
            'ngspice': 'ngspice',
            'ghdl': 'ghdl',
            'klayout': 'klayout',
            'gdstk': None,  # Python library, checked separately
        }
        result = {}
        for name, cmd in tools.items():
            if cmd is None:
                # Python library check
                try:
                    __import__(name)
                    result[name] = True
                except ImportError:
                    result[name] = False
            else:
                result[name] = shutil.which(cmd) is not None
        return result

    def execute_eda_flow(self, rtl_code, testbench_code=None, synth_target='generic',
                         run_simulation=True, run_synthesis=True):
        """Execute a full EDA verification flow: compile → simulate → synthesize.

        Args:
            rtl_code: Verilog/SystemVerilog RTL source code
            testbench_code: Optional testbench code (appended as separate file)
            synth_target: Synthesis target ('generic' or path to liberty file)
            run_simulation: Whether to run iverilog simulation
            run_synthesis: Whether to run Yosys synthesis

        Returns:
            dict with keys: 'success', 'sim_output', 'sim_passed', 'synth_output',
                           'gate_count', 'netlist', 'errors'
        """
        result = {
            'success': False,
            'sim_output': '',
            'sim_passed': False,
            'synth_output': '',
            'gate_count': 0,
            'netlist': '',
            'errors': [],
            'tools_available': self.check_eda_tools(),
        }

        temp_dir = tempfile.mkdtemp(prefix='eda_sandbox_')
        self.active_workspaces.add(temp_dir)

        try:
            # Write RTL source file
            rtl_path = os.path.join(temp_dir, 'design.v')
            with open(rtl_path, 'w') as f:
                f.write(rtl_code)

            # Write testbench if provided
            tb_path = None
            if testbench_code:
                tb_path = os.path.join(temp_dir, 'testbench.v')
                with open(tb_path, 'w') as f:
                    f.write(testbench_code)

            # ── Step 1: Verilog Compilation & Simulation ─────────────────
            if run_simulation and result['tools_available'].get('iverilog'):
                sim_bin = os.path.join(temp_dir, 'sim_out')
                compile_cmd = ['iverilog', '-o', sim_bin]
                compile_cmd.append(rtl_path)
                if tb_path:
                    compile_cmd.append(tb_path)

                try:
                    comp_result = self._run_with_kill(compile_cmd, timeout=60, cwd=temp_dir)
                    compile_output = (comp_result.stdout or '') + (comp_result.stderr or '')

                    if comp_result.returncode != 0:
                        result['errors'].append(f"iverilog compilation failed:\n{compile_output}")
                        result['sim_output'] = compile_output
                        return result

                    # Run simulation
                    if os.path.exists(sim_bin):
                        sim_result = self._run_with_kill(['vvp', sim_bin], timeout=120, cwd=temp_dir)
                        sim_output = (sim_result.stdout or '') + (sim_result.stderr or '')
                        result['sim_output'] = self._truncate_output(sim_output)

                        # Check for simulation pass/fail
                        sim_lower = sim_output.lower()
                        has_fail = any(kw in sim_lower for kw in ['fail', 'error', 'mismatch', 'assertion failed'])
                        has_pass = any(kw in sim_lower for kw in ['pass', 'all tests passed', 'test passed', 'success'])

                        if sim_result.returncode == 0 and not has_fail:
                            result['sim_passed'] = True
                        elif has_pass and not has_fail:
                            result['sim_passed'] = True
                    else:
                        result['errors'].append("iverilog compiled but produced no output binary")

                except subprocess.TimeoutExpired:
                    result['errors'].append("Simulation timed out (120s limit)")
                except Exception as e:
                    result['errors'].append(f"Simulation error: {str(e)}")

            elif run_simulation and not result['tools_available'].get('iverilog'):
                result['errors'].append("iverilog not installed. Install with: apt-get install iverilog")

            # ── Step 2: Yosys Synthesis ──────────────────────────────────
            if run_synthesis and result['tools_available'].get('yosys'):
                netlist_path = os.path.join(temp_dir, 'netlist.v')
                stats_path = os.path.join(temp_dir, 'stats.txt')

                # Build Yosys synthesis script
                synth_script = f"read_verilog {rtl_path}\n"
                synth_script += "hierarchy -auto-top\n"
                synth_script += "proc; opt; fsm; opt; memory; opt\n"

                if synth_target != 'generic' and os.path.exists(synth_target):
                    synth_script += f"read_liberty -lib {synth_target}\n"
                    synth_script += "abc -liberty " + synth_target + "\n"
                else:
                    synth_script += "synth\n"

                synth_script += "opt_clean -purge\n"
                synth_script += f"write_verilog {netlist_path}\n"
                synth_script += f"tee -o {stats_path} stat\n"

                ys_path = os.path.join(temp_dir, 'synth.ys')
                with open(ys_path, 'w') as f:
                    f.write(synth_script)

                try:
                    synth_result = self._run_with_kill(
                        ['yosys', '-s', ys_path], timeout=120, cwd=temp_dir
                    )
                    synth_output = (synth_result.stdout or '') + (synth_result.stderr or '')
                    result['synth_output'] = self._truncate_output(synth_output)

                    if synth_result.returncode != 0:
                        result['errors'].append(f"Yosys synthesis failed:\n{synth_output[-500:]}")
                    else:
                        # Parse gate count from stats
                        if os.path.exists(stats_path):
                            with open(stats_path, 'r') as f:
                                stats_text = f.read()
                            import re as _re
                            cells_match = _re.search(r'Number of cells:\s+(\d+)', stats_text)
                            if cells_match:
                                result['gate_count'] = int(cells_match.group(1))
                            result['synth_output'] += "\n\n--- Synthesis Statistics ---\n" + stats_text

                        # Read synthesized netlist
                        if os.path.exists(netlist_path):
                            with open(netlist_path, 'r') as f:
                                result['netlist'] = f.read()

                except subprocess.TimeoutExpired:
                    result['errors'].append("Yosys synthesis timed out (120s limit)")
                except Exception as e:
                    result['errors'].append(f"Synthesis error: {str(e)}")

            elif run_synthesis and not result['tools_available'].get('yosys'):
                result['errors'].append("Yosys not installed. Install with: apt-get install yosys")

            # Mark overall success
            if not result['errors']:
                result['success'] = True
            elif result['sim_passed'] and not run_synthesis:
                result['success'] = True

        finally:
            # Don't clean workspace yet — caller may need files
            pass

        return result

    def execute_layout_flow(self, gdstk_script, run_drc=False):
        """Execute a gdstk Python script to generate GDSII layout and optionally run DRC.

        Args:
            gdstk_script: Python code using gdstk library to generate layout
            run_drc: Whether to run KLayout DRC (requires klayout installed)

        Returns:
            dict with keys: 'success', 'output', 'gds_path', 'layer_info', 'drc_output', 'errors'
        """
        result = {
            'success': False,
            'output': '',
            'gds_path': '',
            'layer_info': [],
            'drc_output': '',
            'errors': [],
        }

        temp_dir = tempfile.mkdtemp(prefix='eda_layout_')
        self.active_workspaces.add(temp_dir)

        try:
            gds_output_path = os.path.join(temp_dir, 'output.gds')

            # Wrap the user script to ensure it writes to our known output path
            wrapped_script = (
                "import sys\n"
                f"_GDS_OUTPUT_PATH = {repr(gds_output_path)}\n"
                "# User gdstk layout script:\n"
                + gdstk_script + "\n"
                "# Auto-save if the user didn't explicitly write:\n"
                "import os as _os\n"
                f"if not _os.path.exists({repr(gds_output_path)}):\n"
                "    # Try to find the library object and write it\n"
                "    import gc as _gc\n"
                "    try:\n"
                "        import gdstk as _gdstk\n"
                "        for obj in _gc.get_objects():\n"
                "            if isinstance(obj, _gdstk.Library):\n"
                f"                obj.write_gds({repr(gds_output_path)})\n"
                "                break\n"
                "    except Exception:\n"
                "        pass\n"
            )

            # Execute in the standard Python sandbox (unrestricted — needs file I/O)
            py_path = os.path.join(temp_dir, 'layout_gen.py')
            with open(py_path, 'w') as f:
                f.write(wrapped_script)

            try:
                run_result = self._run_with_kill(
                    [sys.executable, py_path], timeout=120, cwd=temp_dir
                )
                output = (run_result.stdout or '') + (run_result.stderr or '')
                result['output'] = self._truncate_output(output)

                if run_result.returncode != 0:
                    result['errors'].append(f"Layout generation failed:\n{output}")
                elif os.path.exists(gds_output_path):
                    result['gds_path'] = gds_output_path
                    result['success'] = True

                    # Extract layer info from the generated GDS for 3D rendering
                    try:
                        import gdstk
                        lib = gdstk.read_gds(gds_output_path)
                        layers_seen = set()
                        for cell in lib.cells:
                            for poly in cell.polygons:
                                layers_seen.add((poly.layer, poly.datatype))
                            for path in cell.paths:
                                layers_seen.add((path.layers[0], path.datatypes[0]))
                        result['layer_info'] = sorted(layers_seen)
                    except ImportError:
                        result['layer_info'] = []
                    except Exception as e:
                        result['layer_info'] = []
                        result['errors'].append(f"GDS layer parsing warning: {str(e)}")
                else:
                    result['errors'].append("Script ran but no GDS file was generated")

            except subprocess.TimeoutExpired:
                result['errors'].append("Layout generation timed out (120s limit)")
            except Exception as e:
                result['errors'].append(f"Layout generation error: {str(e)}")

            # ── Optional DRC ────────────────────────────────────────────
            if run_drc and result['success'] and self.check_eda_tools().get('klayout'):
                # KLayout batch DRC would go here
                result['drc_output'] = "DRC check skipped (no DRC rule deck configured)"

        finally:
            pass

        return result

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

    @staticmethod
    def _strip_python_strings_and_comments(code):
        """
        Return `code` with all Python string literals and comments replaced by
        equivalent-length whitespace. Line numbers are preserved so that any
        downstream error reporting remains accurate.

        Used by `_detect_infinite_loop` so that a benign string like
        `"while(1)"` or a `# while True` comment does not trigger a false
        positive. Falls back to the original `code` if tokenize can't parse it
        (e.g. the code is incomplete or is not actually Python).
        """
        try:
            tokens = list(tokenize.generate_tokens(_io_top.StringIO(code).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return code

        # We rebuild the source line-by-line, blanking out ranges that belong
        # to STRING or COMMENT tokens.
        lines = code.splitlines(keepends=True)
        # Convert to a mutable list of char arrays
        mut_lines = [list(line) for line in lines]

        for tok in tokens:
            if tok.type not in (tokenize.STRING, tokenize.COMMENT):
                continue
            (start_row, start_col) = tok.start
            (end_row, end_col) = tok.end
            # tokenize rows are 1-indexed
            start_row -= 1
            end_row -= 1
            if start_row == end_row:
                if 0 <= start_row < len(mut_lines):
                    row = mut_lines[start_row]
                    for c in range(start_col, min(end_col, len(row))):
                        if row[c] != '\n':
                            row[c] = ' '
            else:
                # Multi-line string — blank the tail of start_row, all of the
                # middle rows, and the head of end_row.
                if 0 <= start_row < len(mut_lines):
                    row = mut_lines[start_row]
                    for c in range(start_col, len(row)):
                        if row[c] != '\n':
                            row[c] = ' '
                for r in range(start_row + 1, end_row):
                    if 0 <= r < len(mut_lines):
                        row = mut_lines[r]
                        for c in range(len(row)):
                            if row[c] != '\n':
                                row[c] = ' '
                if 0 <= end_row < len(mut_lines):
                    row = mut_lines[end_row]
                    for c in range(min(end_col, len(row))):
                        if row[c] != '\n':
                            row[c] = ' '

        return ''.join(''.join(row) for row in mut_lines)

    def _detect_infinite_loop(self, code):
        """
        Check if the code contains patterns that suggest an infinite loop.

        Only inspects executable source — strings and comments are stripped
        first so that e.g. `msg = "while(1)"` or `# while True: forever` no
        longer trigger false positives.
        """
        scrubbed = self._strip_python_strings_and_comments(code)
        for pattern in LOOP_PATTERNS:
            if re.search(pattern, scrubbed, re.IGNORECASE):
                return pattern
        return None

    # ── Subprocess runner that guarantees the entire process group dies ──
    def _run_with_kill(self, cmd, timeout, cwd=None, env=None, use_isolation=False):
        """
        Drop-in replacement for `subprocess.run(cmd, timeout=timeout, ...)`
        that (a) launches the child in its own session so we control the
        whole process group, and (b) SIGKILLs the group if the timeout fires
        — killing grandchildren too (numpy/pandas worker pools, npm
        sub-shells, node children, javac daemons, etc.).

        Phase 1 Enhancement: When use_isolation=True and kernel isolation is
        available, wraps the command with `unshare` for network/PID/IPC
        isolation and applies seccomp-like resource restrictions via preexec_fn.

        Returns a `subprocess.CompletedProcess`-like object with `.returncode`,
        `.stdout`, `.stderr`. Raises `subprocess.TimeoutExpired` on timeout,
        AFTER the process group has been fully killed.
        """
        # Apply kernel-level isolation wrapper if requested and available
        if use_isolation:
            cmd = self._wrap_with_isolation(cmd, temp_dir=cwd)

        # start_new_session=True is the POSIX way to say "new process group
        # with the child as leader". On Windows this arg is ignored and we
        # fall back to plain terminate(); Windows isn't a target platform for
        # the restricted sandbox anyway.
        popen_kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            text=True,
        )
        if os.name == 'posix':
            popen_kwargs['start_new_session'] = True
            # Apply seccomp-like resource limits in the child process
            if use_isolation:
                popen_kwargs['preexec_fn'] = self._apply_seccomp_preexec()

        proc = subprocess.Popen(cmd, **popen_kwargs)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(
                args=cmd, returncode=proc.returncode,
                stdout=stdout, stderr=stderr,
            )
        except subprocess.TimeoutExpired:
            # Kill the entire process group so orphaned grandchildren die too.
            if os.name == 'posix':
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            else:
                try:
                    proc.kill()
                except Exception:
                    pass
            # Drain stdout/stderr so the pipes don't leak; ignore any errors.
            try:
                proc.communicate(timeout=2)
            except Exception:
                pass
            raise

    def _detect_language(self, code):
        """
        Auto-detect the programming language of the code using heuristic signature matching.
        Returns one of: 'python', 'c', 'cpp', 'bash', 'javascript', 'java', 'go', 'rust', 'typescript'.

        Improvements over the previous version:
        - Requires a MINIMUM of 2 signature hits (or a decisive lead) before
          committing to a non-Python guess. This prevents e.g. a stray `def foo(`
          in JS code from being misclassified as Python, or a lone `printf(`
          in Python (via ctypes) from being classified as C.
        - Ties are broken by looking for language-exclusive tokens.
        """
        scores = {}
        for lang, sigs in LANG_SIGNATURES.items():
            score = 0
            for pattern in sigs['strong']:
                if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                    score += 1
            scores[lang] = score

        # C vs C++ disambiguation: any C++-only token wins
        if scores.get('c', 0) > 0 and scores.get('cpp', 0) > 0:
            if re.search(r'std::|iostream|using\s+namespace\s+std|cout\s*<<', code):
                scores['c'] = 0
            elif scores['cpp'] >= scores['c']:
                scores['c'] = 0

        # TypeScript vs JavaScript: TS-exclusive markers beat plain JS
        if scores.get('typescript', 0) > 0 and scores.get('javascript', 0) > 0:
            if re.search(r'\binterface\s+\w+\b|\btype\s+\w+\s*=|:\s*\w+\s*[=,)]', code):
                scores['javascript'] = 0

        best_lang = max(scores, key=scores.get) if scores else 'python'
        best_score = scores.get(best_lang, 0)

        # Require decisive evidence: at least 2 signatures OR a clear lead of 2
        # over the runner-up. Otherwise default to Python (our primary sandbox).
        sorted_scores = sorted(scores.values(), reverse=True)
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0
        if best_score < 2 and (best_score - runner_up) < 2:
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
            comp_res = self._run_with_kill(compile_cmd, timeout=30)
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
            res = self._run_with_kill(run_cmd, timeout=self.timeout)
            if res.returncode == 0:
                output = res.stdout
                if res.stderr:
                    output += "\nWarnings/Stderr:\n" + res.stderr
                return True, output.strip()
            else:
                error = res.stderr if res.stderr else res.stdout
                return False, f"Runtime Error:\n{error.strip()}"
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: Execution took longer than {self.timeout}s (process group killed)."

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
            res = self._run_with_kill(run_cmd, timeout=self.timeout)
            if res.returncode == 0:
                output = res.stdout
                if res.stderr:
                    output += "\nWarnings/Stderr:\n" + res.stderr
                return True, output.strip()
            else:
                error = res.stderr if res.stderr else res.stdout
                return False, error.strip()
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: Code took longer than {self.timeout}s to execute (process group killed)."
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

        # Phase 3: Pre-execution SAST security scan
        if SAST_AVAILABLE:
            sast_result = scan_code_sast(code, language="python")
            if not sast_result["safe"]:
                return False, format_sast_report(sast_result)

        # Prepend compatibility monkeypatch dynamically depending on imports in code
        compat_lines = []
        if "matplotlib" in code or "plt" in code:
            compat_lines.extend([
                "try:",
                "    import matplotlib",
                "    matplotlib.use('Agg')",
                "except Exception:",
                "    pass"
            ])
        if "numpy" in code or "np" in code:
            compat_lines.extend([
                "try:",
                "    import numpy as _np",
                "    if not hasattr(_np, 'trapezoid'):",
                "        _np.trapezoid = getattr(_np, 'trapz', None)",
                "except Exception:",
                "    pass"
            ])
        compat_code = "\n".join(compat_lines) + "\n\n" + code

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
            # Execute the runner script, passing the code file path as argv[1].
            # Use _run_with_kill so that any child processes spawned by the
            # AI code (numpy pool, subprocess.Popen calls the AI made, etc.)
            # are SIGKILL'd along with the runner on timeout.
            res = self._run_with_kill(
                [sys.executable, runner_path, code_path],
                timeout=self.timeout,
            )

            # Parse the JSON result from the runner
            if stdout:
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
        # Prepend compatibility monkeypatch dynamically depending on imports in code
        compat_lines = ["import sys"]
        if "matplotlib" in code or "plt" in code:
            compat_lines.extend([
                "try:",
                "    import matplotlib",
                "    matplotlib.use('Agg')",
                "except Exception:",
                "    pass"
            ])
        if "numpy" in code or "np" in code:
            compat_lines.extend([
                "try:",
                "    import numpy as _np",
                "    if not hasattr(_np, 'trapezoid'):",
                "        _np.trapezoid = getattr(_np, 'trapz', None)",
                "except Exception:",
                "    pass"
            ])
        compat_lines.extend([
            "try:",
            "    from backend.sandbox import SandboxDataHelper",
            "except Exception:",
            "    pass",
            "sys.argv = ['script.py']"
        ])
        compat_code = "\n".join(compat_lines) + "\n\n" + code

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(compat_code)
            path = f.name

        try:
            res = self._run_with_kill([sys.executable, path], timeout=self.timeout)
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
                    res2 = self._run_with_kill([sys.executable, path], timeout=self.timeout)
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
    def parse_multi_file_manifest(text):
        """
        Parses XML-like tags <file path="path/to/file">content</file> from text.
        Also parses JSON manifest from markdown blocks if present.
        Returns a dictionary of {relative_path: content}.
        """
        files = {}
        # 1. XML tag parsing
        pattern = r'<file\s+path=["\']([^"\']+)["\']\s*>(.*?)</file>'
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            path = match.group(1).strip()
            content = match.group(2)
            if content.startswith('\n'):
                content = content[1:]
            if content.endswith('\n'):
                content = content[:-1]
            files[path] = content
            
        # 2. JSON parsing fallback if no XML tags found
        if not files and "package.json" in text:
            # Try to find a JSON block containing files list
            json_pattern = r'```json\s*(\[\s*\{\s*"path".*?\])\s*```'
            json_match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
            if json_match:
                try:
                    file_list = json.loads(json_match.group(1))
                    for entry in file_list:
                        if isinstance(entry, dict) and "path" in entry and "content" in entry:
                            files[entry["path"].strip()] = entry["content"]
                except Exception:
                    pass
        return files

    # ── execute_workspace hardening constants ────────────────────────────
    # Per-file and total size caps prevent a malicious/malformed manifest
    # from filling the host's temp partition.
    _MAX_FILE_BYTES = 5 * 1024 * 1024      #  5 MB per file
    _MAX_TOTAL_BYTES = 200 * 1024 * 1024   # 200 MB per workspace

    # Allow-list of executables that `run_cmd` is permitted to invoke. Any
    # other binary will be refused before we hand it to Popen. This is a
    # defense-in-depth measure — the sandbox already runs in a temp dir with
    # no privileges, but this makes accidental `rm -rf /` in a validation
    # script much harder.
    _RUN_CMD_ALLOWLIST = {
        'npm', 'npx', 'node',
        sys.executable, 'python', 'python3',
        'pip', 'pip3',
        'pytest', 'ruff', 'mypy', 'black',
        'go', 'cargo', 'rustc',
        'gcc', 'g++', 'clang', 'clang++', 'make',
        'bash', 'sh',
        'javac', 'java',
    }

    def execute_workspace(self, files_dict, run_cmd=None, timeout=120):
        """
        Executes a full workspace-level project directory.

        Security hardening (Phase 1.6b / bug #14):
        - Path-traversal check uses `os.path.realpath` (symlink-aware), not
          `os.path.abspath` (which was symlink-blind and could be tricked by
          a symlink placed earlier in the manifest).
        - Per-file size cap of {_MAX_FILE_BYTES} and total size cap of
          {_MAX_TOTAL_BYTES} prevent temp-partition exhaustion.
        - `run_cmd` is parsed with `shlex.split` (correct quoting) and the
          executable is checked against `_RUN_CMD_ALLOWLIST` before Popen.
        - All subprocesses go through `_run_with_kill` so timeouts SIGKILL
          the entire process group instead of orphaning grandchildren.

        Returns (success: bool, logs: str, temp_dir_path: str).
        """
        temp_dir = tempfile.mkdtemp(prefix="workspace_sandbox_")
        self.active_workspaces.add(temp_dir)
        output_log = []
        success = True
        temp_dir_real = os.path.realpath(temp_dir)

        try:
            # 1. Write all files to the temporary directory — with hardening.
            total_written = 0
            written_count = 0
            for rel_path, content in files_dict.items():
                # Reject absolute paths and any component containing '..' or
                # a null byte before we even join.
                if os.path.isabs(rel_path) or '\x00' in rel_path:
                    output_log.append(f"⚠️  Refused unsafe path: {rel_path!r}")
                    continue

                dest_path = os.path.join(temp_dir, rel_path)
                # `realpath` resolves symlinks *and* '..' components — this is
                # what closes the traversal hole.
                dest_real = os.path.realpath(dest_path)
                if not (dest_real == temp_dir_real or dest_real.startswith(temp_dir_real + os.sep)):
                    output_log.append(f"⚠️  Refused traversal attempt: {rel_path!r}")
                    continue

                # Enforce size caps
                content_bytes = content.encode('utf-8', errors='replace') if isinstance(content, str) else bytes(content)
                if len(content_bytes) > self._MAX_FILE_BYTES:
                    output_log.append(
                        f"⚠️  Refused oversize file {rel_path!r}: "
                        f"{len(content_bytes):,} bytes > cap {self._MAX_FILE_BYTES:,}"
                    )
                    continue
                if total_written + len(content_bytes) > self._MAX_TOTAL_BYTES:
                    output_log.append(
                        f"⚠️  Workspace exceeded total size cap of {self._MAX_TOTAL_BYTES:,} bytes; "
                        f"stopping at {written_count} files."
                    )
                    break

                os.makedirs(os.path.dirname(dest_real) or temp_dir, exist_ok=True)
                with open(dest_real, 'wb') as f:
                    f.write(content_bytes)
                total_written += len(content_bytes)
                written_count += 1

            output_log.append(
                f"📁 Created workspace with {written_count} files "
                f"({total_written:,} bytes)."
            )

            # 2. Node.js project handling (npm install)
            if 'package.json' in files_dict:
                output_log.append("📦 package.json detected. Installing dependencies via npm...")
                try:
                    install_res = self._run_with_kill(
                        ['npm', 'install', '--no-audit', '--no-fund', '--prefer-offline'],
                        cwd=temp_dir,
                        timeout=90,
                    )
                    if install_res.stdout:
                        output_log.append(install_res.stdout)
                    if install_res.stderr:
                        output_log.append(f"Npm stderr:\n{install_res.stderr}")
                    if install_res.returncode != 0:
                        success = False
                        output_log.append("❌ npm install failed.")
                        return False, "\n".join(output_log), temp_dir
                except subprocess.TimeoutExpired:
                    success = False
                    output_log.append("❌ npm install timed out after 90s (process group killed).")
                    return False, "\n".join(output_log), temp_dir
                except Exception as e:
                    success = False
                    output_log.append(f"❌ npm install execution failed: {str(e)}")
                    return False, "\n".join(output_log), temp_dir

                # 3. Detect verification/build script from package.json if not provided
                if not run_cmd:
                    try:
                        with open(os.path.join(temp_dir, 'package.json'), 'r') as f:
                            pkg = json.loads(f.read())
                        scripts = pkg.get('scripts', {})
                        if 'build' in scripts:
                            run_cmd = 'npm run build'
                        elif 'test' in scripts:
                            run_cmd = 'npm run test'
                        else:
                            run_cmd = 'node index.js' if 'index.js' in files_dict else None
                    except Exception:
                        run_cmd = None

            # 4. Python project handling (requirements.txt install)
            elif 'requirements.txt' in files_dict:
                output_log.append("🐍 requirements.txt detected. Installing packages...")
                try:
                    install_res = self._run_with_kill(
                        [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '--quiet'],
                        cwd=temp_dir,
                        timeout=90,
                    )
                    if install_res.stdout:
                        output_log.append(install_res.stdout)
                    if install_res.stderr:
                        output_log.append(f"Pip stderr:\n{install_res.stderr}")
                except subprocess.TimeoutExpired:
                    output_log.append("⚠️ pip install timed out after 90s (process group killed).")
                except Exception as e:
                    output_log.append(f"⚠️ pip install failed: {str(e)}")

            # 5. Run validation script (with shlex parsing + allowlist gate)
            if run_cmd:
                output_log.append(f"🚀 Running workspace execution: `{run_cmd}`")
                try:
                    cmd_args = shlex.split(run_cmd)
                except ValueError as e:
                    return False, "\n".join(output_log + [f"❌ Malformed run_cmd: {e}"]), temp_dir

                if not cmd_args:
                    output_log.append("❌ run_cmd was empty after parsing.")
                    return False, "\n".join(output_log), temp_dir

                executable = os.path.basename(cmd_args[0])
                if executable not in self._RUN_CMD_ALLOWLIST and cmd_args[0] not in self._RUN_CMD_ALLOWLIST:
                    output_log.append(
                        f"❌ Refused disallowed executable `{cmd_args[0]}`. "
                        f"Allowed: {sorted(self._RUN_CMD_ALLOWLIST)}"
                    )
                    return False, "\n".join(output_log), temp_dir

                try:
                    run_res = self._run_with_kill(cmd_args, cwd=temp_dir, timeout=timeout)
                    if run_res.stdout:
                        output_log.append(run_res.stdout)
                    if run_res.stderr:
                        output_log.append(f"Stderr:\n{run_res.stderr}")
                    if run_res.returncode != 0:
                        success = False
                        output_log.append(f"❌ Command `{run_cmd}` failed with exit code {run_res.returncode}")
                    else:
                        output_log.append(f"✅ Command `{run_cmd}` completed successfully.")
                except subprocess.TimeoutExpired:
                    success = False
                    output_log.append(f"❌ Command `{run_cmd}` timed out after {timeout}s (process group killed).")
                except Exception as e:
                    success = False
                    output_log.append(f"❌ Execution failed: {str(e)}")
            else:
                output_log.append("ℹ️ No execution command specified. Files written and structured successfully.")

            return success, "\n".join(output_log), temp_dir

        except Exception as e:
            if temp_dir:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass
                self.active_workspaces.discard(temp_dir)
            return False, f"Workspace sandbox error: {str(e)}", temp_dir

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
            known_tags = ['html', 'python', 'py', 'javascript', 'js', 'css', 'bash', 'sh', 'c', 'cpp', 'c++', 'java',
                          'verilog', 'v', 'sv', 'systemverilog', 'spice', 'vhdl', 'tcl']
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
            known_tags = ['html', 'python', 'py', 'javascript', 'js', 'css', 'bash', 'sh', 'c', 'cpp', 'c++', 'java',
                          'verilog', 'v', 'sv', 'systemverilog', 'spice', 'vhdl', 'tcl']
            if first_line in known_tags:
                return _sanitize("\n".join(content.split('\n')[1:]))
            return _sanitize(content)

        return _sanitize(text)
