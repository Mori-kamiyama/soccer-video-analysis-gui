#!/bin/zsh
# swap_to_highfps.sh を取り消して 2fps/1920 構成へ戻す。
set -e
cd "$(dirname "$0")"

if [[ ! -d frames_2fps_backup ]]; then echo "frames_2fps_backup がありません（未切替）。中止。"; exit 1; fi

echo "== ロールバック =="
# フレームを戻す（現 frames= 高fps を frames_hi へ、backupを frames へ）
mv frames frames_hi
mv frames_2fps_backup frames
# box_corrections / player_positions / trim を復元
[[ -f outputs/box_corrections.backup_2fps.csv ]] && mv outputs/box_corrections.backup_2fps.csv outputs/box_corrections.csv
[[ -f outputs/player_positions_all.backup_2fps.csv ]] && mv outputs/player_positions_all.backup_2fps.csv outputs/player_positions_all.csv
[[ -f outputs/trim_settings.backup_2fps.json ]] && mv outputs/trim_settings.backup_2fps.json outputs/trim_settings.json

echo "完了。2fps/1920 構成へ戻しました（frames_hi/ に高fpsフレームは残しています）。"
