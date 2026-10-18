#!/usr/bin/env bash
set -euo pipefail
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="10.176.56.76"
REMOTE_WS="/datadisk1/xxy/AutoSeg3D_v5_stage1_clean_20260604"
mkdir -p "$LOCAL_DIR/remote_76" "$LOCAL_DIR/configs" "$LOCAL_DIR/logs" "$LOCAL_DIR/metrics"
ssh "$HOST" "cd '$REMOTE_WS' && git log -1 --format='%H %ci %s' && echo --- && git status --short" > "$LOCAL_DIR/remote_76/git_status_latest.txt"
ssh "$HOST" "cd '$REMOTE_WS' && find docs_audit -maxdepth 2 -type f -print 2>/dev/null | sort" > "$LOCAL_DIR/remote_76/docs_file_list_latest.txt" || true
rsync -av --include='*/' --include='*.md' --include='*.txt' --exclude='*' "$HOST:$REMOTE_WS/docs_audit/" "$LOCAL_DIR/remote_76/docs_audit/" || true
rsync -av --include='*/' --include='codex_*.py' --exclude='*' "$HOST:$REMOTE_WS/configs/" "$LOCAL_DIR/configs/" || true
