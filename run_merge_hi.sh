#!/bin/zsh
# 選手ID統合エディター（高fps版）の起動ランチャ。
cd "$(dirname "$0")"
PYROOT=/Users/yuta/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none
export TCL_LIBRARY="$PYROOT/lib/tcl8.6"
export TK_LIBRARY="$PYROOT/lib/tk8.6"
exec .venv/bin/python src/player_merge_hi.py "$@"
