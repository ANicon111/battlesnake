"""Open-loop MCTS: plan in a native HISSS game or from a Battlesnake request."""

import argparse, os
from contextlib import closing
from dataclasses import dataclass, field
from math import log, sqrt
from pathlib import Path
import random
from hisss_tools import (
    MOVES,
    advance,
    from_battlesnake,
    replay,
    rollout_moves,
    seed_native,
    serve,
)


@dataclass
class Node:
    parent: object = None
    depth: int = 0
    visits: int = 0
    total: float = 0.0
    children: dict = field(default_factory=dict)  # action -> node


def selection(root, game, horizon, C=1.4):
    """Search through the most promising actions the tree has already included. Return (selected node, prefix reward).

    1. We start the selection at the root with zero prefix reward (=rewards that are gained during the selection phase).
       The supplied game is a simulation copy at the root's board.
       The horizon counts moves from the root.
    2. We keep on selecting new nodes until
        - the game is terminal,
        - the node has reached the horizon,
        - or at least one of the four actions has no child yet, which leaves that action for exploration.
       Otherwise, decide which existing child to visit according to step 3.
    3. Score each child using its mean return plus an exploration bonus:
       UCB = W/N + C × √(ln(P)/N), where W is the child's total return,
       N is its visit count and P is the current parent's visit count.
       Look at the node class at the top of this file. It defines these values.
       Each existing child has already been visited at least once. Choose the highest-scoring child.
       For C we choose 1.4 as a default value, which balances between Exploitation and Exploration.
       This parameter could be further tuned as part of your projects as it may have a strong impact.
    4. Find the action leading to that child in the children dictionary. Apply
       it to the game copy using the function called advance (in gymnasium it is called step,
       but for some reason battlesnake calls it advance).
       It returns (a reward, a bool of the game is done, and an int for if an apple has been collected).
       Add the reward to the prefix reward and continue from the child at step 2.
       Do not change visit counts here. We will do this in the backpropagation phase.
    """
    node = root
    prefix = 0.0
    while node.depth < horizon and not game.is_terminal():
        if len(node.children) < 4:
            break
        parent_visits = node.visits

        def ucb(child):
            if child.visits == 0:
                return float("inf")
            return (child.total / child.visits) + C * sqrt(
                log(parent_visits) / child.visits
            )

        best_action = max(node.children, key=lambda a: ucb(node.children[a]))
        node = node.children[best_action]
        reward, done, _ = advance(game, best_action)
        prefix += reward
        if done:
            break
    return node, prefix


def exploration(node, game, horizon, rng):
    """Expand the selected node once. Return (newly added node, reward for
    this one move to the newly added node).

    The given node represents the one that has been selected by the selection phase.
    It will become the parent of the newly added node.
    The game represents the gamestate at the node provided.

    1. If the current state of the game is terminal or the
       node's depth is equal to the horizon, return that same node and zero.
       No move is made and no new node is added by expansion.
    2. Otherwise, selection stopped because there is an untried action.
       Find any of the actions that have not a corresponding child node (=untried action).
    3. Apply this action to the game. Create one child node with the selected node as its
       parent and a depth one greater than its parent's.
       Store that node as child in the parent's dictionary.
    4. Return the new child and the immediate reward.
    """
    if game.is_terminal() or node.depth == horizon:
        return node, 0.0

    untried = [a for a in rollout_moves(game) if a not in node.children]
    if not untried:
        untried = [a for a in range(4) if a not in node.children]
    if not untried:
        return node, 0.0

    action = rng.choice(untried)
    reward, done, _ = advance(game, action)
    child = Node(parent=node, depth=node.depth + 1)
    node.children[action] = child
    return child, reward


def simulation(game, remaining_horizon, rng):
    """Play out the rest of the simulated future. Return the sum of NEW rewards.
       The game represents the state of the previously generated node of the expansion phase.
       remaining_horizon is the number of time steps we need to simulate.

    1. Start with zero return.
    2. While moves remain and the gamestate is not terminal,
       get all the eligible moves by calling "rollout_moves(game)".
       This function already removes actions that would kill you immediately.
       Use rng.choice() to choose one at random.
    3. Advance the copy by that move
    4. Add its reward to the return and reduce remaining_horizon by 1. Go back to step 2.
    5. After the loop, return the sum.
    """
    total_return = 0.0
    while remaining_horizon > 0 and not game.is_terminal():
        moves = rollout_moves(game)
        if not moves:
            break
        action = rng.choice(moves)
        reward, done, _ = advance(game, action)
        total_return += reward
        remaining_horizon -= 1
        if done:
            break
    return total_return


def backpropagation(node, value):
    """Update the tree's nodes visit and total reward.

    1. Begin at the node returned by exploration.
    2. Record one additional visit at this node and add value to its total.
    3. Use the node's parent as the next node and repeat step 2 until the parent becomes "None"

    This function does not return anything.
    """
    current = node
    while current is not None:
        current.visits += 1
        current.total += value
        current = current.parent


def search(game, iterations, rng, horizon=12):
    """This function implements MCTS search by combining the four functions implemented by you."""
    if iterations < 4 or horizon < 1 or game.is_terminal():
        raise ValueError(
            "Search needs a live board, at least four iterations and a positive horizon."
        )
    root = Node()
    for _ in range(iterations):
        with closing(game.get_copy()) as trial:
            node, prefix = selection(root, trial, horizon)
            leaf, reward = exploration(node, trial, horizon, rng)
            tail = simulation(trial, horizon - leaf.depth, rng)
            backpropagation(leaf, prefix + reward + tail)
    return max(
        root.children, key=lambda a: root.children[a].total / root.children[a].visits
    )


def battlesnake_move(state, iterations, rng, horizon=12):
    with closing(from_battlesnake(state)) as game:
        return MOVES[search(game, iterations, rng, horizon)]


if __name__ == "__main__":
    """This is executed when run from the command line using the following command:
    mcts.py [-h] [--mode {hisss,battlesnake}] [--iterations ITERATIONS] [--seed SEED] [--port PORT]

    to run the agent for battlesnake, use:
    mcts.py --mode battlesnake

    to run the agent using hisss (without hosting it as a server), use:
    mcts.py --mode hisss
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["hisss", "battlesnake"], default="battlesnake"
    )
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    port = int(os.environ.get("PORT", "8000"))
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
    if args.iterations < 4 or args.seed < 0:
        parser.error("Use at least four iterations and a nonnegative seed.")
    rng = random.Random(args.seed)
    seed_native(args.seed)

    if args.mode == "hisss":
        path = Path("results") / f"mcts_seed{args.seed}_iter{args.iterations}.gif"
        replay(
            lambda game: search(game, args.iterations, rng),
            args.seed,
            path,
            f"MCTS: {args.iterations} simulations per move",
        )
    else:
        serve(lambda state: battlesnake_move(state, args.iterations, rng), args.port)
