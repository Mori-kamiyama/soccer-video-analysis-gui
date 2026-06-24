#!/bin/zsh
# 選手IDエディター（画像ベース・結合/カット/追加・1コマ位置合わせ）の起動ランチャ。
cd "$(dirname "$0")"
PYROOT=/Users/yuta/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none
export TCL_LIBRARY="$PYROOT/lib/tcl8.6"
export TK_LIBRARY="$PYROOT/lib/tk8.6"
exec .venv/bin/python src/player_id_editor.py "$@"
