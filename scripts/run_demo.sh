#!/usr/bin/env bash
set -euo pipefail
python -m unittest discover -s tests -v
printf 'Start receiver: python -m src.communication.receiver\n'
printf 'In a second terminal: python -m src.communication.sender_demo\n'
