"""Entry point — run the dashboard.

Run it the same way as your main project: `python app.py` (or just click ▶ Run in
PyCharm on this file). It launches the Streamlit dashboard and opens it in your
browser at http://localhost:8501.

A Streamlit app can't be executed as a plain script, so this wrapper invokes
`streamlit run dashboard_app.py` for you. Needs the [ui] extra:
    pip install -e ".[ui]"
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PORT = "8501"


def main() -> int:
    app = Path(__file__).resolve().parent / "dashboard_app.py"
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Streamlit is not installed. Run:  pip install -e \".[ui]\"")
        return 1
    print(f"Starting the equity-agent dashboard at http://localhost:{PORT}  (Ctrl+C to stop)")
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", PORT],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
