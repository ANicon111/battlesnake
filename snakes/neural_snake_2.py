"""Step 3: use the trained CNN for live Battlesnake /move requests."""

import argparse
import os
import logging
import torch
from neural_snake_2_data.step_0_cnn_state_attributes import (
    ACTIONS,
    CHANNELS,
    INPUT_SHAPE,
    ENCODING_VERSION,
    cnn_state_to_attributes,
)
from neural_snake_2_data.step_2_cnn_train_network import DirectionNetwork
from flask import Flask, request


class CNNAgent:
    def __init__(self, model_path=None):
        torch.set_num_threads(1)
        saved = torch.load(
            (
                model_path
                if model_path is not None
                else "snakes/neural_snake_2_data/models/cnn_global_model.pt"
            ),
            map_location="cpu",
            weights_only=True,
        )
        if (
            saved.get("channels") != CHANNELS
            or tuple(saved.get("input_shape", ())) != INPUT_SHAPE
            or saved.get("actions") != ACTIONS
            or saved.get("encoding_version") != ENCODING_VERSION
        ):
            raise ValueError(
                "Checkpoint and Step 0 encoding differ. Re-encode the data and retrain the global CNN."
            )
        self.recorded_seeds = saved["recorded_seeds"]
        self.model = DirectionNetwork()
        self.model.load_state_dict(saved["weights"])
        self.model.eval()

    def cnn_move(self, state):
        inputs = cnn_state_to_attributes(state).unsqueeze(
            0
        )  # (1,C,H,W), using the recording encoder.
        with torch.inference_mode():
            direction = ACTIONS[int(self.model(inputs).argmax(1).item())]
        # No rule-based correction: evaluate what the network actually learned.
        return {"move": direction}


def cnn_create_app(move, start=None, end=None, info=None):
    """Same four endpoints as the unchanged Exercise 1 server, without reloading."""
    app = Flask("Neural Battlesnake #2")

    @app.get("/")
    def on_info():
        return (
            info()
            if info
            else {
                "apiversion": "1",
                "color": "#004a99",
                "head": "default",
                "tail": "default",
            }
        )

    @app.post("/start")
    def on_start():
        if start:
            start(request.get_json())
        return "ok"

    @app.post("/move")
    def on_move():
        answer = move(request.get_json())
        if not isinstance(answer, dict) or answer.get("move") not in ACTIONS:
            raise ValueError("Agent must return {'move': 'up'/'down'/'left'/'right'}")
        return answer

    @app.post("/end")
    def on_end():
        if end:
            end(request.get_json())
        return "ok"

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    port = int(os.environ.get("PORT", "8000"))
    agent = CNNAgent()
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(f"CNN agent: http://127.0.0.1:{port}")
    cnn_create_app(agent.cnn_move).run(host="127.0.0.1", port=port, debug=False)
