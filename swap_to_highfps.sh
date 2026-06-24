#!/bin/zsh
# 高fps(12fps/2560)版へ本番ディレクトリを切り替える。
# 旧2fps構成はすべてバックアップしてから置換する（rollback_highfps.sh で復元可）。
# 実行前提: frames_hi/ と outputs/box_corrections_hi.csv が出来ていること。
set -e
cd "$(dirname "$0")"

if [[ ! -d frames_hi ]]; then echo "frames_hi/ がありません"; exit 1; fi
if [[ ! -f outputs/box_corrections_hi.csv ]]; then echo "box_corrections_hi.csv がありません"; exit 1; fi
if [[ -d frames_2fps_backup ]]; then echo "既に切替済み（frames_2fps_backup あり）。中止。"; exit 1; fi

echo "== バックアップ＆切替 =="
# フレーム
mv frames frames_2fps_backup
mv frames_hi frames
# box_corrections
cp outputs/box_corrections.csv outputs/box_corrections.backup_2fps.csv
cp outputs/box_corrections_hi.csv outputs/box_corrections.csv
# player_positions_all（基底検出）→ 退避し、ヘッダのみの空CSVに（add行だけで成立）
cp outputs/player_positions_all.csv outputs/player_positions_all.backup_2fps.csv
head -1 outputs/player_positions_all.backup_2fps.csv > outputs/player_positions_all.csv
# トリムは新番号で全範囲に（注釈のある使用区間のみ自然に表示される）
[[ -f outputs/trim_settings.json ]] && cp outputs/trim_settings.json outputs/trim_settings.backup_2fps.json
echo '{"start": 1, "end": 999999, "last_idx": 0}' > outputs/trim_settings.json
# pitch_points.json は1920基準のまま（変更なし）

echo "完了。frames/ は12fps/2560、box_corrections.csv は移行版になりました。"
echo "  ./run_box_gui.sh で確認、visualize_offside は --start 1267 --end 7861 --skip 5725-6055 --fps 12 --inbetween 1"
