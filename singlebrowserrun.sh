#!/bin/sh
[ -z "$B_WIDTH" ] && export B_WIDTH=11
[ -z "$B_HEIGHT" ] && export B_HEIGHT=11
cd -P -- "$(dirname -- "$0")"
.venv/bin/python main.py&
PID=$!
sleep 3
battlesnake/battlesnake play -W $B_WIDTH -H $B_HEIGHT --name s1 --url http://localhost:8000 -g solo --browser
sleep 10
kill $PID