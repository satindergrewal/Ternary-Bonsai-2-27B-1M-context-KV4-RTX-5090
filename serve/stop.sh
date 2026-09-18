#!/bin/bash
pkill -f llama-server
sleep 2
if pgrep -f llama-server >/dev/null; then echo "STILL-RUNNING"; exit 1; fi
echo "STOPPED"
