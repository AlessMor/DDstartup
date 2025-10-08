"""
Inputs package initialization

This file adds the ddstartup directory to the Python path,
allowing config files to import from utils module.
"""
import sys
from pathlib import Path

# Add ddstartup directory to Python path
ddstartup_dir = Path(__file__).parent.parent / "ddstartup"
if ddstartup_dir.exists() and str(ddstartup_dir) not in sys.path:
    sys.path.insert(0, str(ddstartup_dir))
