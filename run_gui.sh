#!/bin/zsh
# 選手トラッキング割当・訂正GUIの起動ランチャ
# uv製venvのpythonはTcl/Tkのパスが通っていないので環境変数で繋ぐ。
cd "$(dirname "$0")"
PYROOT=/Users/yuta/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none
export TCL_LIBRARY="$PYROOT/lib/tcl8.6"
export TK_LIBRARY="$PYROOT/lib/tk8.6"
exec .venv/bin/python src/player_tracker_gui.py "$@"
