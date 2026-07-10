"""
EDA Tool Setup Utility for DeepThink AIOS Chip Design Pipeline.

Checks for installed open-source EDA tools and provides installation
commands for missing dependencies.

Supported tools:
- iverilog (Icarus Verilog) — RTL simulation
- yosys — Logic synthesis
- ngspice — SPICE circuit simulation
- gdstk — GDSII physical layout (Python library)
- klayout — Physical verification / DRC (optional)
"""

import subprocess
import sys
import shutil
import os


def check_tools():
    """Check which EDA tools are available on this system.
    
    Returns:
        dict: Tool name → bool (True if available)
    """
    cli_tools = {
        'iverilog': 'iverilog',
        'vvp': 'vvp',
        'yosys': 'yosys',
        'ngspice': 'ngspice',
        'ghdl': 'ghdl',
        'klayout': 'klayout',
    }
    python_libs = ['gdstk', 'gdspy']
    
    result = {}
    
    # Check CLI tools
    for name, cmd in cli_tools.items():
        result[name] = shutil.which(cmd) is not None
    
    # Check Python libraries
    for lib in python_libs:
        try:
            __import__(lib)
            result[lib] = True
        except ImportError:
            result[lib] = False
    
    return result


def get_install_instructions():
    """Return human-readable installation instructions for each missing tool."""
    tools = check_tools()
    instructions = []
    
    if not tools.get('iverilog') or not tools.get('vvp'):
        instructions.append(
            "📦 **Icarus Verilog** (RTL Simulation):\n"
            "   Ubuntu/Debian: sudo apt-get install iverilog\n"
            "   macOS:         brew install icarus-verilog\n"
            "   Kaggle/Colab:  !apt-get install -y iverilog"
        )
    
    if not tools.get('yosys'):
        instructions.append(
            "📦 **Yosys** (Logic Synthesis):\n"
            "   Ubuntu/Debian: sudo apt-get install yosys\n"
            "   macOS:         brew install yosys\n"
            "   Kaggle/Colab:  !apt-get install -y yosys"
        )
    
    if not tools.get('ngspice'):
        instructions.append(
            "📦 **Ngspice** (SPICE Simulation):\n"
            "   Ubuntu/Debian: sudo apt-get install ngspice\n"
            "   macOS:         brew install ngspice\n"
            "   Kaggle/Colab:  !apt-get install -y ngspice"
        )
    
    if not tools.get('gdstk'):
        instructions.append(
            "📦 **gdstk** (GDSII Layout Generation):\n"
            "   All platforms:  pip install gdstk"
        )
    
    if not tools.get('klayout'):
        instructions.append(
            "📦 **KLayout** (Physical Verification — Optional):\n"
            "   Ubuntu/Debian: sudo apt-get install klayout\n"
            "   macOS:         brew install klayout\n"
            "   Note: KLayout is optional. DRC will be skipped without it."
        )
    
    return instructions


def install_tools(tools_to_install=None):
    """Attempt to install missing EDA tools.
    
    Best-effort installation — works on Kaggle/Colab where root access is available.
    On local machines, prints instructions instead.
    
    Args:
        tools_to_install: List of tool names to install, or None for all missing
        
    Returns:
        dict: Tool name → install result message
    """
    current = check_tools()
    results = {}
    
    if tools_to_install is None:
        tools_to_install = [name for name, available in current.items() if not available]
    
    # Apt packages
    apt_map = {
        'iverilog': 'iverilog',
        'vvp': 'iverilog',  # Same package
        'yosys': 'yosys',
        'ngspice': 'ngspice',
        'klayout': 'klayout',
        'ghdl': 'ghdl',
    }
    
    # Pip packages
    pip_map = {
        'gdstk': 'gdstk',
        'gdspy': 'gdspy',
    }
    
    # Try apt-get for CLI tools
    apt_packages = set()
    for tool in tools_to_install:
        if tool in apt_map and not current.get(tool):
            apt_packages.add(apt_map[tool])
    
    if apt_packages:
        # Check if we have apt-get
        if shutil.which('apt-get'):
            try:
                # Try to install (requires root — works on Kaggle/Colab)
                cmd = ['apt-get', 'install', '-y'] + sorted(apt_packages)
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    for pkg in apt_packages:
                        results[pkg] = "✅ Installed successfully"
                else:
                    for pkg in apt_packages:
                        results[pkg] = f"❌ apt-get failed: {result.stderr[:200]}"
            except subprocess.TimeoutExpired:
                for pkg in apt_packages:
                    results[pkg] = "❌ Installation timed out"
            except PermissionError:
                for pkg in apt_packages:
                    results[pkg] = f"❌ Permission denied — run: sudo apt-get install {pkg}"
        else:
            for pkg in apt_packages:
                results[pkg] = "❌ apt-get not available on this system"
    
    # Try pip for Python libraries
    for tool in tools_to_install:
        if tool in pip_map and not current.get(tool):
            pip_pkg = pip_map[tool]
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', pip_pkg,
                     '--quiet', '--disable-pip-version-check'],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    results[pip_pkg] = "✅ Installed successfully"
                else:
                    results[pip_pkg] = f"❌ pip install failed: {result.stderr[:200]}"
            except Exception as e:
                results[pip_pkg] = f"❌ Error: {str(e)}"
    
    return results


def print_status():
    """Print a formatted status report of all EDA tools."""
    tools = check_tools()
    
    print("\n╔══════════════════════════════════════════════════╗")
    print("║       DeepThink AIOS — EDA Tool Status          ║")
    print("╠══════════════════════════════════════════════════╣")
    
    categories = {
        'RTL Simulation': [('iverilog', 'Icarus Verilog'), ('vvp', 'VVP Runner')],
        'Logic Synthesis': [('yosys', 'Yosys Open Synthesis')],
        'SPICE Simulation': [('ngspice', 'Ngspice')],
        'Physical Layout': [('gdstk', 'GDSTK (Python)'), ('gdspy', 'GDSpy (Python)')],
        'Verification': [('klayout', 'KLayout DRC'), ('ghdl', 'GHDL (VHDL)')],
    }
    
    for category, tool_list in categories.items():
        print(f"║  {category}:")
        for key, label in tool_list:
            status = "✅" if tools.get(key) else "❌"
            print(f"║    {status} {label:<30} {'Available' if tools.get(key) else 'Not Found'}")
    
    print("╚══════════════════════════════════════════════════╝")
    
    missing = get_install_instructions()
    if missing:
        print("\n⚠️  Missing tools — install with:")
        for instr in missing:
            print(f"\n{instr}")
    else:
        print("\n✅ All EDA tools are installed and ready!")


if __name__ == "__main__":
    print_status()
