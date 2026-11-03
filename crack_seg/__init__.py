import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure conda env Library/bin DLLs (OpenBLAS, OpenMP, etc.) are discoverable on Windows
if sys.platform == "win32":
    lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(lib_bin):
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib_bin)
            except Exception:
                pass
        os.environ["PATH"] = lib_bin + os.pathsep + os.environ.get("PATH", "")
