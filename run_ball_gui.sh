#!/bin/zsh
# ボール・アノテーションGUIの起動ランチャ。
# uv製venvのpythonはTcl/Tkのパスが通っていないので環境変数で繋ぐ。
cd "$(dirname "$0")"
PYROOT=/Users/yuta/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none
export TCL_LIBRARY="$PYROOT/lib/tcl8.6"
export TK_LIBRARY="$PYROOT/lib/tk8.6"
# 既定は高解像度の frames_ball。--frames-dir で上書き可（argparseは後勝ち）。
exec .venv/bin/python src/ball_annotator.py --frames-dir frames_ball "$@"
