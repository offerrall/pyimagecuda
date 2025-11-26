import re
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent


def find_project_root(start: Path) -> Path:
    cur = start
    while cur != cur.parent:
        if (cur / "src").exists():
            return cur
        cur = cur.parent
    return start

ROOT = find_project_root(SCRIPT_DIR)
SRC_DIR = ROOT / "src"

print(f"[generate] Script path: {THIS_FILE}")
print(f"[generate] Project root: {ROOT}")
print(f"[generate] Source dir: {SRC_DIR}")

PAT = re.compile(r"\bPyObject\s*\*\s*([A-Za-z_]\w*)\s*\(")
SKIP_SUFFIXES = ("_internal",)

funcs_by_dir: dict[str, dict[str, set[str]]] = {}

for cu_file in SRC_DIR.rglob("*.cu"):
    text = cu_file.read_text("utf-8", "ignore")
    for fn in PAT.findall(text):
        if any(fn.endswith(suf) for suf in SKIP_SUFFIXES):
            continue

        rel_dir = cu_file.parent.relative_to(SRC_DIR).as_posix() or "."
        fname   = cu_file.name

        funcs_by_dir.setdefault(rel_dir, {}).setdefault(fname, set()).add(fn)

include_path = SRC_DIR / "include.h"

inc_lines = [
    "#pragma once",
    "#include <Python.h>",
    "#include <cuda_runtime.h>",
    "#include <stdint.h>",
    "",
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    "",
]

for category in sorted(funcs_by_dir):
    inc_lines.append(f"// {category}")
    for cu_file in sorted(funcs_by_dir[category]):
        inc_lines.append(f"// {cu_file}")
        for fn in sorted(funcs_by_dir[category][cu_file]):
            inc_lines.append(f"PyObject* {fn}(PyObject* self, PyObject* args);")
        inc_lines.append("")

inc_lines += [
    "",
    "#ifdef __cplusplus",
    "}",
    "#endif",
    "",
]

include_path.write_text("\n".join(inc_lines), "utf-8")
print(f"[generate] Wrote {include_path}")

wrapper_path = SRC_DIR / "python_wrapper.cpp"

wrap_lines = [
    "#include <Python.h>",
    "#include \"include.h\"",
    "",
    "static PyMethodDef all_methods[] = {",
]

for category in sorted(funcs_by_dir):
    if category != ".":
        wrap_lines.append(f"    // {category}")
    for cu_file in sorted(funcs_by_dir[category]):
        wrap_lines.append(f"    // {cu_file}")
        for fn in sorted(funcs_by_dir[category][cu_file]):
            public_name = fn[3:] if fn.startswith("py_") else fn
            desc = " ".join(word.capitalize() for word in public_name.split("_"))
            wrap_lines.append(
                f'    {{"{public_name}", {fn}, METH_VARARGS, "{desc}"}},'
            )
        wrap_lines.append("")
    wrap_lines.append("")

wrap_lines += [
    "    {NULL, NULL, 0, NULL}",
    "};",
    "",
    "static struct PyModuleDef module = {",
    "    PyModuleDef_HEAD_INIT,",
    '    "pyimagecuda_internal",',
    '    "CUDA image processing internal module",',
    "    -1,",
    "    all_methods",
    "};",
    "",
    "PyMODINIT_FUNC PyInit_pyimagecuda_internal(void) {",
    "    return PyModule_Create(&module);",
    "}",
    "",
]

wrapper_path.write_text("\n".join(wrap_lines), "utf-8")
print(f"[generate] Wrote {wrapper_path}")

total = sum(len(funcs) for files in funcs_by_dir.values() for funcs in files.values())
print(f"[generate] Found {len(funcs_by_dir)} directories with {total} functions.")
