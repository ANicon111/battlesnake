#!/usr/bin/env python3
import os, subprocess, sys, time
from pathlib import Path

os.chdir(Path(__file__).parent.resolve())

W = os.getenv("B_WIDTH", "11")
H = os.getenv("B_HEIGHT", "11")
procs, cli_args = [], []

# Filter out directories and only keep files directly in the 'snakes' folder
snake_files = sorted([p for p in Path("snakes").iterdir() if p.is_file()])

try:
    for i, p in enumerate(snake_files):
        port = 8001 + i
        cmd = [sys.executable, str(p)] if p.suffix == ".py" else [str(p)]

        procs.append(subprocess.Popen(cmd, env={**os.environ, "PORT": str(port)}))
        cli_args.extend(["--name", p.stem, "--url", f"http://localhost:{port}"])
        print(f"Started '{p.stem}' on port {port}")

    time.sleep(3)
    subprocess.run(
        ["battlesnake/battlesnake", "play", "-W", W, "-H", H, "-v", *cli_args]
    )

finally:
    print("\nShutting down snake processes...")
    for p in procs:
        p.terminate()
        p.wait()
