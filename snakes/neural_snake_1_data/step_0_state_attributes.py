"""Shared Battlesnake attributes and training examples, independent of any agent.

Edit state_to_attributes() to select recorded features and INPUT_COLUMNS to
select numeric network inputs. Recording, training and live inference all use
this schema. Other agents can import these helpers without importing a bot or
starting the simulator.
"""

import copy
from pathlib import Path

import pandas as pd

ACTIONS = ["up", "down", "left", "right"]
DELTAS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

# These numeric columns are the network's inputs. The complete apple list and
# original state are also recorded, but variable-length lists are not MLP inputs.
# Select numeric network inputs
INPUT_COLUMNS = [
    "board_width",
    "board_height",
    "health",
    "body_length",
    "head_x",
    "head_y",
    "safe_up",
    "safe_down",
    "safe_left",
    "safe_right",
    "dist_closest_food",
    "food_dx",
    "food_dy",
]


def state_to_attributes(state):
    """Extracts state attributes into a fixed-length numeric dictionary."""
    board = state["board"]
    snake = state["you"]

    # Basic metadata
    attributes = {
        "board_width": float(board["width"]),
        "board_height": float(board["height"]),
        "health": float(snake["health"]),
        "body_length": float(len(snake["body"])),
        "head_x": float(snake["head"]["x"]),
        "head_y": float(snake["head"]["y"]),
    }

    head = snake["head"]

    # Collect all occupied tile coordinates (bodies of all snakes)
    occupied_coords = set()
    for current_snake in board["snakes"]:
        for part in current_snake["body"]:
            occupied_coords.add((part["x"], part["y"]))

    # Function to check adjacent tile safety
    def is_safe(x, y):
        if x < 0 or x >= board["width"] or y < 0 or y >= board["height"]:
            return 0.0  # Wall collision
        if (x, y) in occupied_coords:
            return 0.0  # Body collision
        return 1.0

    attributes["safe_up"] = is_safe(head["x"], head["y"] + 1)
    attributes["safe_down"] = is_safe(head["x"], head["y"] - 1)
    attributes["safe_left"] = is_safe(head["x"] - 1, head["y"])
    attributes["safe_right"] = is_safe(head["x"] + 1, head["y"])

    # Nearest Food Attributes (Manhattan distance)
    foods = board.get("food", [])
    if foods:
        closest_food = min(
            foods,
            key=lambda f: abs(f["x"] - head["x"]) + abs(f["y"] - head["y"]),
        )
        attributes["dist_closest_food"] = float(
            abs(closest_food["x"] - head["x"]) + abs(closest_food["y"] - head["y"])
        )
        attributes["food_dx"] = float(closest_food["x"] - head["x"])
        attributes["food_dy"] = float(closest_food["y"] - head["y"])
    else:
        # Default fallback values if no food is present on board
        attributes["dist_closest_food"] = float(board["width"] + board["height"])
        attributes["food_dx"] = 0.0
        attributes["food_dy"] = 0.0

    return attributes


def make_training_example(game_state, direction, *, label_source, seed=None):
    """Build an independent state/action snapshot for any agent's dataset.

    The caller owns the rows and decides when to store them. For example:
    rows.append(make_training_example(state, direction, label_source="my_agent"))
    """
    row = {
        **state_to_attributes(game_state),
        "game_id": game_state["game"]["id"],
        "turn": game_state["turn"],
        "seed": seed,
        "direction": direction,
        "label_source": label_source,
        "state": game_state,
    }
    return copy.deepcopy(row)


def load_dataframe(path):
    """Restore the saved pandas DataFrame, including its nested apple lists."""
    df = pd.read_json(Path(path), orient="table")
    required = set(INPUT_COLUMNS + ["game_id", "turn", "direction", "state"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. Record a new dataset after changing attributes."
        )
    if df.empty:
        raise ValueError(
            "The DataFrame is empty. Record some agent decisions in Step 1 first."
        )
    if not df["direction"].isin(ACTIONS).all():
        raise ValueError("Every label must be up, down, left or right.")
    if df.duplicated(["game_id", "turn"]).any():
        raise ValueError("The dataset contains duplicate game/turn pairs.")
    return df
