"""
pdk_setup.py — Process Design Kit (PDK) Toolchain Detector & Downloader
========================================================================
Provides detection, download, and validation of open-source semiconductor
Process Design Kits (PDKs) for the DeepThink AIOS EDA pipeline.

Currently supported PDKs:
    - SkyWater 130nm (SKY130) — sky130_fd_sc_hd standard cells
    - GlobalFoundries 180nm (GF180MCU) — planned

This module follows the same pattern as eda_setup.py for consistency.
"""

import os
import shutil
import json

# ─────────────────────────────────────────────────────────────────────────
# PDK installation base directory
# ─────────────────────────────────────────────────────────────────────────
PDK_BASE_DIR = os.path.expanduser("~/.aios/pdk")

# ─────────────────────────────────────────────────────────────────────────
# PDK Definitions
# ─────────────────────────────────────────────────────────────────────────
PDK_DEFINITIONS = {
    "sky130": {
        "name": "SkyWater 130nm (SKY130)",
        "description": "Open-source 130nm CMOS process from SkyWater Technology / Google",
        "liberty_file": "sky130_fd_sc_hd__tt_025C_1v80.lib",
        "tech_lef": "sky130_fd_sc_hd.tlef",
        "cell_library": "sky130_fd_sc_hd",
        "github_repo": "google/skywater-pdk-libs-sky130_fd_sc_hd",
        "install_dir": os.path.join(PDK_BASE_DIR, "sky130"),
        "min_size_bytes": 1000,  # Minimum valid liberty file size
    },
    "gf180": {
        "name": "GlobalFoundries 180nm (GF180MCU)",
        "description": "Open-source 180nm CMOS process from GlobalFoundries",
        "liberty_file": "gf180mcu_fd_sc_mcu7t5v0__tt_025C_1v80.lib",
        "tech_lef": "gf180mcu_fd_sc_mcu7t5v0.tlef",
        "cell_library": "gf180mcu_fd_sc_mcu7t5v0",
        "github_repo": "google/gf180mcu-pdk",
        "install_dir": os.path.join(PDK_BASE_DIR, "gf180"),
        "min_size_bytes": 1000,
    },
}

# ─────────────────────────────────────────────────────────────────────────
# PDK Detection
# ─────────────────────────────────────────────────────────────────────────

def detect_pdk(pdk_name="sky130"):
    """Check if a PDK is installed and its liberty file is valid.

    Args:
        pdk_name: Name of the PDK (e.g., 'sky130', 'gf180')

    Returns:
        dict with keys: 'available' (bool), 'liberty_path' (str or None),
        'message' (str)
    """
    if pdk_name not in PDK_DEFINITIONS:
        return {
            "available": False,
            "liberty_path": None,
            "message": f"Unknown PDK: {pdk_name}. Supported: {list(PDK_DEFINITIONS.keys())}"
        }

    pdk = PDK_DEFINITIONS[pdk_name]
    liberty_path = os.path.join(pdk["install_dir"], pdk["liberty_file"])

    if not os.path.exists(liberty_path):
        return {
            "available": False,
            "liberty_path": None,
            "message": f"{pdk['name']} not found at {liberty_path}. "
                       f"Run `python backend/pdk_setup.py` to set up."
        }

    # Validate file size (basic integrity check)
    file_size = os.path.getsize(liberty_path)
    if file_size < pdk["min_size_bytes"]:
        return {
            "available": False,
            "liberty_path": None,
            "message": f"{pdk['name']} liberty file appears corrupted "
                       f"({file_size} bytes). Please re-download."
        }

    return {
        "available": True,
        "liberty_path": liberty_path,
        "message": f"{pdk['name']} detected at {liberty_path} ({file_size:,} bytes)"
    }


def get_pdk_liberty_path(pdk_name="sky130"):
    """Get the absolute path to the PDK liberty file, or None if not available."""
    result = detect_pdk(pdk_name)
    return result["liberty_path"] if result["available"] else None


def get_pdk_yosys_commands(pdk_name="sky130"):
    """Generate Yosys TCL synthesis commands for a specific PDK.

    Returns a list of Yosys commands that map the design to real standard cells.
    Returns None if the PDK is not available (falls back to generic synthesis).
    """
    liberty_path = get_pdk_liberty_path(pdk_name)
    if not liberty_path:
        return None

    return [
        f"dfflibmap -liberty {liberty_path}",
        f"abc -liberty {liberty_path}",
        "clean",
    ]


def generate_drc_script(pdk_name="sky130", gds_input="output.gds", report_output="drc_report.xml"):
    """Generate a KLayout DRC script for the specified PDK.

    Args:
        pdk_name: Name of the PDK
        gds_input: Input GDS file path
        report_output: Output DRC report path (XML format)

    Returns:
        str: The DRC script content, or None if PDK not available
    """
    if pdk_name == "sky130":
        return f'''# SKY130 Basic DRC Rules (subset for AIOS validation)
# These rules check the most critical design rules for the SKY130 process.

source('{gds_input}')

# Metal1 minimum width: 0.14um
m1 = input(68, 20)
m1.width(0.14).output("M1.W.1", "Metal1 minimum width violation (0.14um)")

# Metal1 minimum spacing: 0.14um
m1.space(0.14).output("M1.S.1", "Metal1 minimum spacing violation (0.14um)")

# Via1 minimum size: 0.15um x 0.15um
via1 = input(68, 44)
via1.width(0.15).output("VIA1.W.1", "Via1 minimum width violation (0.15um)")

# Metal2 minimum width: 0.14um
m2 = input(69, 20)
m2.width(0.14).output("M2.W.1", "Metal2 minimum width violation (0.14um)")

# Metal2 minimum spacing: 0.14um
m2.space(0.14).output("M2.S.1", "Metal2 minimum spacing violation (0.14um)")

# Poly minimum width: 0.15um
poly = input(66, 20)
poly.width(0.15).output("POLY.W.1", "Poly minimum width violation (0.15um)")

# Diffusion minimum width: 0.15um
diff = input(65, 20)
diff.width(0.15).output("DIFF.W.1", "Diffusion minimum width violation (0.15um)")
'''
    return None


def parse_drc_report(report_path):
    """Parse a KLayout DRC XML report and extract violations.

    Args:
        report_path: Path to the DRC report XML file

    Returns:
        list of dicts with keys: 'rule', 'message', 'count', 'coordinates'
    """
    violations = []
    if not os.path.exists(report_path):
        return violations

    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(report_path)
        root = tree.getroot()

        for category in root.findall('.//category'):
            rule_name = category.findtext('name', 'Unknown')
            description = category.findtext('description', '')
            items = category.findall('.//item')
            coords = []
            for item in items:
                # Extract coordinate values from the item
                values = item.findall('.//value')
                for val in values:
                    text = val.text or ''
                    if text.strip():
                        coords.append(text.strip())

            if items:
                violations.append({
                    "rule": rule_name,
                    "message": description,
                    "count": len(items),
                    "coordinates": coords[:5],  # Limit to first 5 for readability
                })
    except Exception as e:
        violations.append({
            "rule": "PARSE_ERROR",
            "message": f"Failed to parse DRC report: {str(e)}",
            "count": 0,
            "coordinates": [],
        })

    return violations


def format_drc_violations_for_llm(violations):
    """Format DRC violations into a human-readable string for LLM feedback.

    Args:
        violations: List of violation dicts from parse_drc_report()

    Returns:
        str: Formatted violation summary for the LLM prompt
    """
    if not violations:
        return "✅ DRC PASSED: No design rule violations detected."

    lines = ["❌ DRC VIOLATIONS DETECTED:\n"]
    for v in violations:
        lines.append(f"  Rule: {v['rule']}")
        lines.append(f"  Issue: {v['message']}")
        lines.append(f"  Violations: {v['count']}")
        if v['coordinates']:
            lines.append(f"  Locations: {', '.join(v['coordinates'][:3])}")
        lines.append("")

    lines.append("Please fix the layout code to resolve these violations.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# PDK Status Report
# ─────────────────────────────────────────────────────────────────────────

def get_pdk_status():
    """Get the status of all supported PDKs.

    Returns:
        dict mapping PDK name to detection result
    """
    status = {}
    for pdk_name in PDK_DEFINITIONS:
        status[pdk_name] = detect_pdk(pdk_name)
    return status


# ─────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DeepThink AIOS — PDK Status Report")
    print("=" * 60)
    print()

    status = get_pdk_status()
    for pdk_name, result in status.items():
        icon = "✅" if result["available"] else "❌"
        print(f"  {icon} {PDK_DEFINITIONS[pdk_name]['name']}")
        print(f"     {result['message']}")
        print()

    # Check for KLayout
    klayout_path = shutil.which("klayout")
    if klayout_path:
        print(f"  ✅ KLayout found at: {klayout_path}")
    else:
        print("  ⚠️  KLayout not found. DRC verification will be skipped.")
        print("     Install: sudo apt-get install klayout")
    print()

    print("=" * 60)
    print("  PDK Installation Directory:", PDK_BASE_DIR)
    print("=" * 60)
""",
<br>"""
