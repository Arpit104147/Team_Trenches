"""
security.py — Enterprise Security Module for DeepThink AIOS
============================================================
Provides:
    1. Pre-execution SAST (Static Application Security Testing) scanning
    2. JWT token creation and verification for API authentication
    3. Air-gap mode configuration for offline-only deployments

All features gracefully degrade — if dependencies are missing or auth
is not configured, the system continues to function without security
enforcement (suitable for local single-user development).
"""

import os
import re
import time
import json
import hashlib
import hmac
import base64

# ─────────────────────────────────────────────────────────────────────────
# SAST (Static Application Security Testing)
# ─────────────────────────────────────────────────────────────────────────

# High-severity patterns that should BLOCK execution
PYTHON_BLOCK_PATTERNS = [
    # Command injection via shell=True
    (r'subprocess\.\w+\([^)]*shell\s*=\s*True', "Command injection: subprocess with shell=True"),
    # Direct system command execution
    (r'os\.system\s*\(', "Command injection: os.system() call"),
    # Eval/exec on user-controlled input
    (r'eval\s*\(\s*input\s*\(', "Code injection: eval(input())"),
    (r'exec\s*\(\s*input\s*\(', "Code injection: exec(input())"),
    # Reverse shell patterns
    (r'socket\.socket\([^)]*\).*connect\s*\(', "Network: Possible reverse shell"),
    # File system destruction
    (r'shutil\.rmtree\s*\(\s*["\']/', "Filesystem: Recursive delete from root"),
    (r'os\.remove\s*\(\s*["\']/', "Filesystem: Delete from root path"),
    # Pickle deserialization (arbitrary code execution)
    (r'pickle\.loads?\s*\(', "Deserialization: pickle.load() can execute arbitrary code"),
    # Known exfiltration techniques
    (r'requests\.(get|post)\s*\(\s*["\']https?://', "Network: HTTP request to external URL"),
    (r'urllib\.request\.urlopen', "Network: URL open to external resource"),
]

# Medium-severity patterns that generate WARNINGS but allow execution
PYTHON_WARN_PATTERNS = [
    (r'__import__\s*\(', "Dynamic import: __import__() bypasses static analysis"),
    (r'compile\s*\([^)]+,\s*["\']exec["\']\s*\)', "Dynamic code: compile() with exec mode"),
    (r'ctypes\.\w+', "Low-level: ctypes usage can bypass Python safety"),
    (r'importlib\.import_module', "Dynamic import: importlib can load arbitrary modules"),
]

# Patterns for other languages
VERILOG_BLOCK_PATTERNS = [
    (r'\$system\s*\(', "SystemVerilog: $system() call can execute shell commands"),
    (r'\$fopen\s*\(\s*["\']/', "Verilog: File open from root path"),
]

C_BLOCK_PATTERNS = [
    (r'system\s*\(', "C: system() call executes shell commands"),
    (r'popen\s*\(', "C: popen() opens a process pipe"),
    (r'execve?\s*\(', "C: exec() replaces current process"),
    (r'fork\s*\(\s*\)', "C: fork() creates child process"),
    (r'unlink\s*\(\s*["\']/', "C: unlink() deleting from root path"),
]


def scan_code_sast(code, language="python"):
    """Perform static security analysis on code before sandbox execution.

    Args:
        code: Source code string to analyze
        language: Programming language ('python', 'verilog', 'c', 'cpp', etc.)

    Returns:
        dict with keys:
            'safe' (bool): True if no blocking issues found
            'blocked' (list): High-severity issues that block execution
            'warnings' (list): Medium-severity issues (informational)
            'score' (int): Security risk score (0=safe, 100=critical)
    """
    result = {
        "safe": True,
        "blocked": [],
        "warnings": [],
        "score": 0,
    }

    if not code or not code.strip():
        return result

    # Select patterns based on language
    block_patterns = []
    warn_patterns = []

    if language in ("python", "py"):
        block_patterns = PYTHON_BLOCK_PATTERNS
        warn_patterns = PYTHON_WARN_PATTERNS
    elif language in ("verilog", "systemverilog", "sv"):
        block_patterns = VERILOG_BLOCK_PATTERNS
    elif language in ("c", "cpp", "c++"):
        block_patterns = C_BLOCK_PATTERNS

    # Scan for blocking patterns
    for pattern, description in block_patterns:
        matches = re.findall(pattern, code, re.IGNORECASE | re.MULTILINE)
        if matches:
            result["blocked"].append({
                "pattern": pattern,
                "description": description,
                "occurrences": len(matches),
            })
            result["score"] += 30 * len(matches)

    # Scan for warning patterns
    for pattern, description in warn_patterns:
        matches = re.findall(pattern, code, re.IGNORECASE | re.MULTILINE)
        if matches:
            result["warnings"].append({
                "pattern": pattern,
                "description": description,
                "occurrences": len(matches),
            })
            result["score"] += 10 * len(matches)

    # Cap score at 100
    result["score"] = min(result["score"], 100)
    result["safe"] = len(result["blocked"]) == 0

    return result


def format_sast_report(scan_result):
    """Format SAST scan results into a human-readable security report.

    Args:
        scan_result: Result dict from scan_code_sast()

    Returns:
        str: Formatted security report
    """
    if scan_result["safe"] and not scan_result["warnings"]:
        return "🛡️ SAST: Code passed security analysis."

    lines = []

    if not scan_result["safe"]:
        lines.append("🚨 **SECURITY ALERT — Code Blocked**\n")
        lines.append("The following high-severity security issues were detected:\n")
        for issue in scan_result["blocked"]:
            lines.append(f"  ❌ {issue['description']} ({issue['occurrences']} occurrence(s))")
        lines.append("\n⛔ Execution blocked for safety. Please remove the flagged patterns.")

    if scan_result["warnings"]:
        lines.append("\n⚠️ **Security Warnings** (execution allowed):\n")
        for issue in scan_result["warnings"]:
            lines.append(f"  ⚠️ {issue['description']} ({issue['occurrences']} occurrence(s))")

    lines.append(f"\n📊 Risk Score: {scan_result['score']}/100")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# JWT Authentication (Lightweight, no external dependency required)
# ─────────────────────────────────────────────────────────────────────────

# JWT secret — generated at startup or from environment variable
JWT_SECRET = os.environ.get("AIOS_JWT_SECRET", hashlib.sha256(
    f"aios-default-{os.getpid()}".encode()
).hexdigest())
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.environ.get("AIOS_JWT_EXPIRY_HOURS", "24"))


def _base64url_encode(data):
    """Base64url encode bytes."""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')


def _base64url_decode(s):
    """Base64url decode string."""
    s += '=' * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)


def create_jwt_token(user_id, extra_claims=None):
    """Create a JWT token for API authentication.

    Args:
        user_id: Unique user identifier
        extra_claims: Optional dict of additional claims

    Returns:
        str: Signed JWT token string
    """
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    payload = {
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + (JWT_EXPIRY_HOURS * 3600),
    }
    if extra_claims:
        payload.update(extra_claims)

    header_b64 = _base64url_encode(json.dumps(header, separators=(',', ':')).encode())
    payload_b64 = _base64url_encode(json.dumps(payload, separators=(',', ':')).encode())
    signing_input = f"{header_b64}.{payload_b64}"

    signature = hmac.new(
        JWT_SECRET.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    signature_b64 = _base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def verify_jwt_token(token):
    """Verify a JWT token and return the payload.

    Args:
        token: JWT token string

    Returns:
        dict: Decoded payload if valid, or None if invalid/expired
    """
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}"
        expected_signature = hmac.new(
            JWT_SECRET.encode(), signing_input.encode(), hashlib.sha256
        ).digest()
        actual_signature = _base64url_decode(signature_b64)

        if not hmac.compare_digest(expected_signature, actual_signature):
            return None

        # Decode payload
        payload = json.loads(_base64url_decode(payload_b64))

        # Check expiration
        if payload.get("exp", 0) < time.time():
            return None

        return payload

    except Exception:
        return None


def get_user_id_from_token(token):
    """Extract user_id from a verified JWT token.

    Args:
        token: JWT token string

    Returns:
        str: User ID if token is valid, or 'anonymous' if not
    """
    payload = verify_jwt_token(token)
    return payload.get("sub", "anonymous") if payload else "anonymous"


# ─────────────────────────────────────────────────────────────────────────
# Air-Gap Configuration
# ─────────────────────────────────────────────────────────────────────────

class AirGapConfig:
    """Configuration for air-gapped (offline-only) deployments.

    When air-gap mode is active, the system:
    - Disables all web search / DuckDuckGo scraping
    - Disables model auto-download from HuggingFace Hub
    - Disables external API calls
    - Uses only locally installed PDK files and model weights
    """

    def __init__(self):
        self.enabled = os.environ.get("AIOS_AIR_GAP", "0") == "1"
        if self.enabled:
            print("🔒 Air-Gap Mode: ACTIVE — All outbound network features disabled.")

    @property
    def allow_web_search(self):
        return not self.enabled

    @property
    def allow_model_download(self):
        return not self.enabled

    @property
    def allow_external_api(self):
        return not self.enabled

    def to_dict(self):
        return {
            "air_gap_enabled": self.enabled,
            "web_search": self.allow_web_search,
            "model_download": self.allow_model_download,
            "external_api": self.allow_external_api,
        }


# Global air-gap config instance
air_gap = AirGapConfig()


# ─────────────────────────────────────────────────────────────────────────
# Auth Status
# ─────────────────────────────────────────────────────────────────────────

AUTH_ENABLED = os.environ.get("AIOS_AUTH_ENABLED", "0") == "1"

if AUTH_ENABLED:
    print("🔑 Authentication: ENABLED — JWT tokens required for API access.")
else:
    print("🔓 Authentication: DISABLED — Open access (local development mode).")
