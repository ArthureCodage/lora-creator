#!/usr/bin/env bash
# Lanceur Linux de LoRA Creator.
# Python de kohya_ss : par défaut ~/kohya_ss/venv, sinon définir KOHYA_DIR (ou "python" dans config.json).
# Options transmises au serveur, ex. :  ./lancer-lora-creator.sh --host 0.0.0.0 --port 7870 --no-browser
set -e
KOHYA_DIR="${KOHYA_DIR:-$HOME/kohya_ss}"
PY="$KOHYA_DIR/venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "kohya_ss introuvable : $PY"
  echo "Installe kohya_ss ou définis KOHYA_DIR (ex. KOHYA_DIR=/workspace/kohya_ss $0)."
  exit 1
fi
cd "$(dirname "$(readlink -f "$0")")/app"
exec "$PY" server.py "$@"
