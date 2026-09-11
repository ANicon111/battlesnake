# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import typing


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
        "color": "#ff0000",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    x = my_head["x"]
    y = my_head["y"]
    height = game_state["board"]["height"]
    width = game_state["board"]["width"]

    if x == 0 and y == 0:
        return {"move": "up"}

    if y == 0:
        return {"move": "left"}

    if width % 2 == 1:
        if (height - y) % 2 == 1:
            if x == (width - 2):
                return {"move": "right"}
            if x == (width - 1):
                return {"move": "down"}
        if (height - y) % 2 == 0:
            if x == (width - 1):
                return {"move": "left"}
            if x == (width - 2):
                return {"move": "down"}

    if x == (width - 1):
        return {"move": "down"}

    if x % 2 == 0 and y == (height - 1):
        return {"move": "right"}

    if x % 2 == 0:
        return {"move": "up"}

    if x % 2 == 1 and y == 1:
        return {"move": "right"}

    if x % 2 == 1:
        return {"move": "down"}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
