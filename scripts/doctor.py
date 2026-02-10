# -*- coding: utf-8 -*-
import sys
from pathlib import Path

def ok(msg): print(f"[OK] {msg}")
def warn(msg): print(f"[WARN] {msg}")
def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def main():
    # repo root = scripts/doctor.py 的上一级
    root = Path(__file__).resolve().parents[1]
    ok(f"Repo root: {root}")

    # 1) Python version
    if sys.version_info < (3, 10):
        fail(f"Python {sys.version.split()[0]} is too old. Use Python 3.10+")
    ok(f"Python: {sys.version.split()[0]}")

    # 2) Required files
    req = root / "requirements.txt"
    if not req.exists():
        fail("requirements.txt not found (run from repo root)")
    ok("requirements.txt exists")

    entry = root / "app" / "main.py"
    if not entry.exists():
        fail("app/main.py not found (run from repo root)")
    ok("app/main.py exists")

    # 3) Model file (primary + fallback)
    primary_model = root / "app" / "ml" / "obesity_model.joblib"
    if primary_model.exists():
        ok(f"Model exists: {primary_model.relative_to(root)}")
    else:
        warn("Primary model not found: app/ml/obesity_model.joblib")
        warn("Hint: generate it by running: python app/ml/train_obesity_model.py")

        candidates = list(root.rglob("*.joblib"))
        if candidates:
            ok("Found other .joblib file(s): " + ", ".join(str(p.relative_to(root)) for p in candidates[:3]))
        else:
            warn("No .joblib model found in repo.")

    # 4) DB file check (should NOT be tracked)
    db = root / "elora.db"
    if db.exists():
        warn("elora.db exists locally (OK). Make sure it is ignored by git.")
    else:
        ok("No local elora.db found (it will be created at runtime).")

    ok("Doctor check finished.")

if __name__ == "__main__":
    main()
