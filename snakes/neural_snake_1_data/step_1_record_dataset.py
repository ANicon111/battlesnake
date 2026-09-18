"""Step 1: run X complete games with the regular Battlesnake simulator.

Example: python step_1_record_dataset.py --games 10
The agent records its own state/action examples; this file only runs games
and saves the collected rows as a pandas DataFrame.
"""

import argparse
import logging
from pathlib import Path
import random
from flask import Flask, request
import subprocess
import threading

import pandas as pd
from werkzeug.serving import make_server

import main as agent
from step_0_state_attributes import load_dataframe


def record_games(
    games=10,
    data="data/rule_moves.json",
    engine="battlesnake/battlesnake",
    seed=100,
    seconds=120,
):
    """Start the ordinary agent server, then call `battlesnake play` X times."""
    if games < 1 or seconds <= 0:
        raise ValueError("Game count and time limit must be positive.")
    engine, path = Path(engine).resolve(), Path(data)
    if not engine.is_file():
        raise FileNotFoundError(f"Battlesnake simulator not found: {engine}")
    df = load_dataframe(path) if path.exists() else pd.DataFrame()
    if len(df) and "seed" in df:
        seed = max(seed, int(df.seed.max()) + 1)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse the regular Battlesnake API server from the starter project.
    handlers = {name: getattr(agent, name) for name in ("info", "start", "move", "end")}
    app = Flask("Battlesnake")

    @app.get("/")
    def on_info():
        return handlers["info"]()

    @app.post("/start")
    def on_start():
        game_state = request.get_json()
        handlers["start"](game_state)
        return "ok"

    @app.post("/move")
    def on_move():
        game_state = request.get_json()
        return handlers["move"](game_state)

    @app.post("/end")
    def on_end():
        game_state = request.get_json()
        handlers["end"](game_state)
        return "ok"

    @app.after_request
    def identify_server(response):
        response.headers.set("server", "battlesnake/github/starter-snake-python")
        return response

    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    agent.recorded_rows.clear()
    agent.recording_enabled = True
    thread.start()

    def save_dataset():
        nonlocal df
        if not agent.recorded_rows:
            return
        df = pd.concat([df, pd.DataFrame(agent.recorded_rows)], ignore_index=True)
        df = df.drop_duplicates(["game_id", "turn"], keep="last").reset_index(drop=True)
        temporary = path.with_suffix(".tmp")
        df.to_json(temporary, orient="table", index=False, indent=2)
        temporary.replace(path)
        df.drop(columns=["state"], errors="ignore").to_csv(
            path.with_suffix(".csv"), index=False
        )
        agent.recorded_rows.clear()

    try:
        for number in range(games):
            agent.recording_seed = seed + number
            random.seed(agent.recording_seed)
            print(f"Game {number + 1}/{games}, seed {agent.recording_seed}", flush=True)
            command = [
                str(engine),
                "play",
                "-g",
                "solo",
                "-W",
                "11",
                "-H",
                "11",
                "--seed",
                str(agent.recording_seed),
                "--timeout",
                "500",
                "--name",
                "Rule-based agent",
                "--url",
                f"http://127.0.0.1:{server.server_port}",
            ]
            # The game ends naturally. A timeout aborts the batch, not a fake win/loss.
            with path.with_suffix(".engine.log").open("a", encoding="utf8") as log:
                subprocess.run(
                    command, check=True, timeout=seconds, stdout=log, stderr=log
                )
            save_dataset()
            print(f"Saved {len(df)} decisions to {path}", flush=True)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        agent.recording_enabled = False
        agent.recording_seed = None
        save_dataset()  # Preserve collected rows even if interrupted mid-game.
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--data", default="data/rule_moves.json")
    p.add_argument("--engine", default="battlesnake/battlesnake")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument(
        "--seconds",
        type=float,
        default=120,
        help="Maximum seconds per game before aborting",
    )
    args = p.parse_args()
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    try:
        record_games(args.games, args.data, args.engine, args.seed, args.seconds)
    except KeyboardInterrupt:
        print("Stopped. Collected examples have been saved.")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        p.exit(1, f"Recording failed: {error}. See the dataset's .engine.log file.\n")


if __name__ == "__main__":
    main()
