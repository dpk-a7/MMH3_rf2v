import ctypes
import os
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path


class RuntimeJitTests(unittest.TestCase):
    def test_python_extension_loads_from_runtime_cache(self) -> None:
        cache_dir = os.environ.get("TRITON_CACHE_DIR")
        with tempfile.TemporaryDirectory(
            prefix="mmh3-jit-test-", dir=cache_dir
        ) as directory:
            library = Path(directory) / "probe.so"
            subprocess.run(
                [
                    os.environ.get("CC", "gcc"),
                    "-x",
                    "c",
                    "-shared",
                    "-fPIC",
                    f"-I{sysconfig.get_path('include')}",
                    "-o",
                    str(library),
                    "-",
                ],
                input=(
                    "#include <Python.h>\n"
                    "const char *mmh3_python_version(void) { return Py_GetVersion(); }\n"
                ),
                text=True,
                check=True,
                timeout=30,
            )

            module = ctypes.PyDLL(str(library))
            module.mmh3_python_version.restype = ctypes.c_char_p

            self.assertEqual(module.mmh3_python_version().decode(), sys.version)


if __name__ == "__main__":
    unittest.main()
