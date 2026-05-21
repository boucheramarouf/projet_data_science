"""
Helper script to launch the dashboard or the API.

Usage:
    python run.py dashboard     # Launch Streamlit dashboard (port 8501)
    python run.py api           # Launch FastAPI server (port 8000)
    python run.py train         # Re-train all models
"""

import sys, os, subprocess

cmd = sys.argv[1] if len(sys.argv) > 1 else "dashboard"

if cmd == "dashboard":
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    os.path.join("dashboard", "app.py"), "--server.port", "8501"])
elif cmd == "api":
    subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app",
                    "--reload", "--port", "8000"])
elif cmd == "train":
    subprocess.run([sys.executable, "train.py"])
else:
    print(f"Unknown command '{cmd}'. Use: dashboard | api | train")
    sys.exit(1)
