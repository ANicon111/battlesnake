"""Step 0: encode the whole board as a (channels, height, width) tensor.

Recording and live inference use the same function. Add new channels here,
then re-encode the saved states (or record again) and retrain after changes.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import neural_snake_2_data.cnn_config as config

ACTIONS = ["up", "down", "left", "right"]
BOARD_WIDTH = BOARD_HEIGHT = 11
CHANNELS = ["my_head", "my_body", "food", "health", "board"]
INPUT_SHAPE = (len(CHANNELS), BOARD_HEIGHT, BOARD_WIDTH)
ENCODING_VERSION = 3  # Whole-board view, without opponents or hazards.


def cnn_state_to_attributes(state):
    """Return float32 input[channel, y, x] in absolute board coordinates.

    Occupancy planes contain 0/1. Health is repeated over the board as health/100.
    Up increases y; use origin='lower' when plotting this top-down board view.
    The full 11x11 board is included, with no crop or head-relative translation.
    The board plane contains 1, distinguishing board cells from zero padding.
    All current body cells, including the head and tail, are marked as occupied.
    """
    board, snake = state["board"], state["you"]
    board_w, board_h = board["width"], board["height"]

    # Target observation dimensions (assumed 11x11 from INPUT_SHAPE)
    obs_h, obs_w = INPUT_SHAPE[1], INPUT_SHAPE[2]  # e.g., 11, 11

    # Determine top-left/bottom-left crop offset
    if board_w > obs_w or board_h > obs_h:
        head_x, head_y = snake["body"][0]["x"], snake["body"][0]["y"]
        # Center the window around the head
        offset_x = head_x - obs_w // 2
        offset_y = head_y - obs_h // 2
    else:
        offset_x, offset_y = 0, 0

    inputs = torch.zeros(INPUT_SHAPE, dtype=torch.float32)

    def cnn_mark(channel, positions):
        ch_idx = CHANNELS.index(channel)
        for p in positions:
            # Translate coordinates relative to the crop window
            rx = p["x"] - offset_x
            ry = p["y"] - offset_y
            # Only mark positions that fit inside the local observation window
            if 0 <= rx < obs_w and 0 <= ry < obs_h:
                inputs[ch_idx, ry, rx] = 1

    # Mark entity positions
    cnn_mark("my_head", snake["body"][:1])
    cnn_mark("my_body", snake["body"])
    cnn_mark("food", board["food"])

    # Mark health channel across the observation tensor
    inputs[CHANNELS.index("health")].fill_(snake["health"] / 100.0)

    # Mark board channel only where valid board tiles exist in the cropped window
    board_ch_idx = CHANNELS.index("board")
    for ry in range(obs_h):
        for rx in range(obs_w):
            orig_x = rx + offset_x
            orig_y = ry + offset_y
            if 0 <= orig_x < board_w and 0 <= orig_y < board_h:
                inputs[board_ch_idx, ry, rx] = 1.0

    return inputs


def cnn_make_training_example(game_state, direction, *, label_source, seed=None):
    if direction not in ACTIONS:
        raise ValueError(f"Unknown direction: {direction}")
    return copy.deepcopy(
        {
            # JSON stores nested lists; Step 2 restores a float32 tensor.
            "tensor": cnn_state_to_attributes(game_state).tolist(),
            "channels": CHANNELS.copy(),
            "encoding_version": ENCODING_VERSION,
            "game_id": game_state["game"]["id"],
            "turn": game_state["turn"],
            "seed": seed,
            "direction": direction,
            "label_source": label_source,
            "state": game_state,
        }
    )


def cnn_dataframe_to_tensors(df):
    """Stack DataFrame examples into (N,C,H,W), with integer action labels."""
    required = {
        "tensor",
        "channels",
        "encoding_version",
        "game_id",
        "turn",
        "direction",
        "state",
    }
    if required - set(df.columns):
        raise ValueError("This is not a CNN dataset. Record a new dataset with Step 1.")
    if df.empty:
        raise ValueError("Record some decisions in Step 1 first.")
    if not df.direction.isin(ACTIONS).all() or df.duplicated(["game_id", "turn"]).any():
        raise ValueError("Invalid direction labels or duplicate game/turn pairs.")
    if (
        not df.channels.apply(lambda names: list(names) == CHANNELS).all()
        or not (df.encoding_version == ENCODING_VERSION).all()
    ):
        raise ValueError(
            "Recorded encoding differs from Step 0. Re-encode the saved states or record a new dataset."
        )
    arrays = np.asarray(df["tensor"].tolist(), dtype=np.float32)
    if arrays.shape != (len(df), *INPUT_SHAPE) or not np.isfinite(arrays).all():
        raise ValueError(f"Expected finite inputs shaped (N, {INPUT_SHAPE}).")
    x = torch.from_numpy(arrays)
    y = torch.tensor(df.direction.map(ACTIONS.index).to_numpy(), dtype=torch.long)
    return x, y


def cnn_load_dataframe(path):
    df = pd.read_json(Path(path), orient="table")
    cnn_dataframe_to_tensors(
        df
    )  # Refuse incompatible encodings before appending/training.
    return df


def cnn_reencode_dataframe(df):
    """Reuse the exact states, labels and game split; never call the teacher again."""
    if "state" not in df:
        raise ValueError("Re-encoding requires the original game state in every row.")
    result = df.copy(deep=True)
    result["tensor"] = [cnn_state_to_attributes(state).tolist() for state in df.state]
    result["channels"] = [CHANNELS.copy() for _ in range(len(df))]
    result["encoding_version"] = ENCODING_VERSION
    cnn_dataframe_to_tensors(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-encode saved states without playing new games."
    )
    parser.add_argument(
        "--source", default=config.PROJECT_DIR / "data" / "rule_tensor_moves.json"
    )
    parser.add_argument("--output", default=config.DATA_PATH)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        parser.error(f"{path} already exists. Choose a new --output filename.")
    df = cnn_reencode_dataframe(pd.read_json(args.source, orient="table"))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="table", index=False, indent=2)
    df.drop(columns=["state", "tensor", "channels"]).to_csv(
        path.with_suffix(".csv"), index=False
    )
    print(f"Re-encoded {len(df)} decisions into {path}. Original data is unchanged.")
