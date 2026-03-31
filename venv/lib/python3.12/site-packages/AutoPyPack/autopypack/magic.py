from IPython.core.magic import register_cell_magic
import tempfile
import os
from . import autopypack as autopypack_module  # Access AutoPyPack class

@register_cell_magic
def autopypack(line, cell):
    """Cell magic that auto-installs packages used in the cell code."""
    print("[AutoPyPack] 📦 Scanning cell for imports...")

    # Write cell contents to a temporary file and scan it
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(cell)
        tmp_path = tmp.name

    try:
        autopypack_module.scan_file(tmp_path)
    finally:
        os.remove(tmp_path)
