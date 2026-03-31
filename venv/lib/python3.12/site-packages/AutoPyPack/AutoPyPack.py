# AutoPyPack.py

from .autopypack import AutoPyPack, autopypack
import sys
import os
import traceback
import inspect

__version__ = "1.2.0"
__all__ = ['AutoPyPack', 'autopypack']

# Re-export key methods so user can call them directly:
# import AutoPyPack
# AutoPyPack.install()
install = AutoPyPack.install
scan_file = AutoPyPack.scan_file

DEBUG = False

def _auto_scan():
    """Automatically scan the importing script for package dependencies."""
    if _in_notebook():
        return  
    try:
        for frame_info in inspect.stack():
            if frame_info.filename.endswith(('AutoPyPack.py', '__init__.py')) or \
               '<frozen' in frame_info.filename or \
               frame_info.filename.startswith('<'):
                continue
            if os.path.isfile(frame_info.filename):
                print(f"[AutoPyPack] Auto-scanning imports in {os.path.basename(frame_info.filename)}")
                autopypack.scan_file(frame_info.filename)
                break
    except Exception as e:
        print(f"[AutoPyPack] Error while auto-scanning: {e}")
        traceback.print_exc()

# Trigger scan when this file is the main entry point
if __name__ == "__main__":
    _auto_scan()

# ---------------------------------------
# Jupyter notebook support: %%autopypack
# ---------------------------------------

def _in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell",)  # Jupyter notebook/lab
    except:
        return False

if _in_notebook():
    try:
        from .autopypack import magic  # triggers @register_cell_magic
        print("[AutoPyPack] ✅ Magic command %%autopypack loaded!")
    except Exception as e:
        print(f"[AutoPyPack] ❌ Failed to load magic command: {e}")
