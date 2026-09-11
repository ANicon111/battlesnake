#!/bin/sh
[ -z "$B_WIDTH" ] && export B_WIDTH=11
[ -z "$B_HEIGHT" ] && export B_HEIGHT=11
cd -P -- "$(dirname -- "$0")"
tmux new-session '.venv/bin/python main.py' \; setw -g mouse on \; split-window -h "sleep 3;battlesnake/battlesnake play -W $B_WIDTH -H $B_HEIGHT --name s1 --url http://localhost:8000 -g solo -v;sleep 1000"