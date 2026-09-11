#!/bin/sh
cd -P -- "$(dirname -- "$0")"
tmux new-session  '.venv/bin/python main.py' \; setw -g mouse on \; split-window -h 'sleep 3;battlesnake/battlesnake play -W 11 -H 11 --name s1 --url http://localhost:8000 -g solo -v;sleep 1000'