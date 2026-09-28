"""
rlm_utf8_patch.py
=================
Fixes an encoding bug in the `rlm` library that makes non-ASCII documents
unusable on Windows.

The bug
-------
Every REPL environment hands the document to the sandbox by writing it to a
temp file and generating code to read it back:

    with open(context_path, "w") as f:        # host side
        f.write(context_payload)
    self.execute_code("with open(r'...', 'r') as f: context = f.read()")

Neither call passes `encoding`, so both use the platform default. On Linux and
macOS that is UTF-8 and nothing goes wrong. On Windows it is cp1252, and any
document containing a character outside that codepage raises

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u25e6'

Real documents hit this constantly: FinDVer's SEC filings use bullet glyphs
(U+25E6, U+25CF), and typographic quotes and dashes appear throughout the
corpora. The failure is total -- the instance is lost, not degraded.

Why patch rather than set PYTHONUTF8=1
--------------------------------------
`PYTHONUTF8=1` does fix it, but it has to be set before the interpreter starts,
so it silently does nothing if a caller forgets, and it changes the default
encoding for every file the process touches rather than the two that are wrong.
This patch targets exactly the affected methods and travels with the repo, so a
fresh `pip install rlm` on any machine stays fixed.

Both halves must be corrected: writing UTF-8 while the generated read code still
uses cp1252 produces mojibake instead of an exception, which is worse.

Usage
-----
    import rlm_utf8_patch; rlm_utf8_patch.apply()

Idempotent, and safe if a future release fixes this upstream: it verifies the
unencoded `open(` is actually present before replacing anything, and reports
which environments it touched.
"""

import io
import json
import os


_APPLIED = False


def _make_add_context(cls_name):
    """Build a replacement `add_context` that is explicit about encoding on
    both the host write and the REPL-side read."""

    def add_context(self, context_payload, context_index=None):
        if context_index is None:
            context_index = self._context_count
        var_name = f"context_{context_index}"

        if isinstance(context_payload, str):
            context_path = os.path.join(self.temp_dir,
                                        f"context_{context_index}.txt")
            with io.open(context_path, "w", encoding="utf-8") as f:
                f.write(context_payload)
            code = (f"with open(r'{context_path}', 'r', encoding='utf-8') "
                    f"as _rlm_f:\n    {var_name} = _rlm_f.read()")
        else:
            context_path = os.path.join(self.temp_dir,
                                        f"context_{context_index}.json")
            with io.open(context_path, "w", encoding="utf-8") as f:
                # ensure_ascii=False keeps the file readable and byte-identical
                # to the text the model will see; the encoding is explicit now,
                # so escaping is no longer load-bearing.
                json.dump(context_payload, f, ensure_ascii=False)
            code = ("import json as _rlm_json\n"
                    f"with open(r'{context_path}', 'r', encoding='utf-8') "
                    f"as _rlm_f:\n    {var_name} = _rlm_json.load(_rlm_f)")

        self.execute_code(code)
        if context_index == 0:
            self.execute_code(f"context = {var_name}")
        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    add_context.__doc__ = (f"UTF-8-safe replacement for {cls_name}.add_context "
                           f"(see rlm_utf8_patch).")
    add_context._rlm_utf8_patched = True
    return add_context


def _needs_patch(cls):
    """True if this class's add_context still opens files without an encoding.

    Checked against the source rather than assumed, so an upstream fix is left
    alone instead of being overwritten by this shim.
    """
    if getattr(cls.add_context, "_rlm_utf8_patched", False):
        return False
    try:
        import inspect
        src = inspect.getsource(cls.add_context)
    except Exception:
        return True          # cannot read it; patching is the safe default
    return 'encoding="utf-8"' not in src and "encoding='utf-8'" not in src


def apply(verbose=True):
    """Patch every REPL environment whose add_context lacks an encoding.

    Returns the list of class names patched.
    """
    global _APPLIED
    if _APPLIED:
        return []

    targets = []
    for mod_name, cls_name in (("rlm.environments.local_repl", "LocalREPL"),
                               ("rlm.environments.ipython_repl", "IPythonREPL"),
                               ("rlm.environments.docker_repl", "DockerREPL")):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception:
            continue          # optional backend not installed
        if not hasattr(cls, "add_context"):
            continue
        if _needs_patch(cls):
            cls.add_context = _make_add_context(cls_name)
            targets.append(cls_name)

    _APPLIED = True
    if verbose and targets:
        print(f"[rlm_utf8_patch] UTF-8 encoding enforced on: "
              f"{', '.join(targets)}")
    elif verbose:
        print("[rlm_utf8_patch] nothing to patch (already UTF-8 safe)")
    return targets


if __name__ == "__main__":
    # Self-test: round-trip a document containing the characters that broke
    # FinDVer, through a real LocalREPL.
    apply()
    from rlm.environments.local_repl import LocalREPL
    probe = "bullets \u25e6 \u25cf quotes \u201cx\u201d dash \u2014 end"
    env = LocalREPL()
    try:
        env.load_context(probe)
        # Compare inside the REPL and print a plain verdict: the library's
        # REPLResult.__repr__ raises (it reads a field named llm_calls that the
        # dataclass actually calls rlm_calls), so the object is never formatted.
        res = env.execute_code(
            "print('MATCH' if context == " + ascii(probe) + " else 'MISMATCH')")
        out = str(getattr(res, "output", "") or getattr(res, "stdout", "") or "")
        good = "MATCH" in out and "MISMATCH" not in out
        print("round-trip:", "OK" if good else "MISMATCH")
        print("  sent:", ascii(probe))
        print("  repl:", out.strip()[:120])
    finally:
        try:
            env.close()
        except Exception:
            pass
