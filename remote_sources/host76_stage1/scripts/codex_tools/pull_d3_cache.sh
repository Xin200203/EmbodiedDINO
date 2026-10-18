#!/usr/bin/env bash
log(){ echo "[$(date -u "+%F %T UTC")] $*"; }
log "waiting for 184 D3_EXTRACT_DONE"
while ! ssh -o ConnectTimeout=10 -o BatchMode=yes nebula@10.177.69.184 'test -f /home/nebula/xxy/feat_weights/D3_EXTRACT_DONE' 2>/dev/null; do sleep 300; done
log "184 extraction done; rsync starting"
mkdir -p /datadisk4/xxy/dino_cache
rsync -a nebula@10.177.69.184:/home/nebula/xxy/dinov3_cache_tmp/dinov3_vitl16_416x560_fp16 /datadisk4/xxy/dino_cache/
rc=$?
n=$(find /datadisk4/xxy/dino_cache/dinov3_vitl16_416x560_fp16 -name '*.pt' | wc -l)
log "rsync rc=$rc files=$n"
if [ "$rc" -eq 0 ] && [ "$n" -ge 108662 ]; then
  touch /datadisk4/xxy/dino_cache/D3_CACHE_READY
  log "gate opened"
else
  log "PULL INCOMPLETE"
fi
