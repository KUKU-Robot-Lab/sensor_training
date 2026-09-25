#!/usr/bin/env bash
# 5090(로컬) ↔ arm4090 모델 산출물 양방향 동기화 — 어느 쪽에서든 실행 가능.
#
# ★이름 규약 주의: arm4090 은 UI 콤보 폭 때문에 SATS run 의 ecomesh_ 접두사를 제거해
#   두 PC 의 폴더명이 다르다(코드는 두 이름 모두 인식). 그래서 통째 rsync 는 같은 run 이
#   두 이름으로 중복 생기므로 금지 — **이름을 정규화해 개별 동기화**한다.
#   전송 방향은 "없는 쪽에 채워주기"(--ignore-existing). 재학습으로 갱신하는 경우는
#   같은 이름이 이미 있으므로 손대지 않는다 → 그때는 수동으로 지우고 다시 실행.
set -u
REMOTE="${1:-user@arm4090}"
LOCAL_ROOT="$HOME/sensor_training"
REMOTE_ROOT="~/sensor_training"

remote_name() {                       # 원격 실제 폴더명(접두사 유무 흡수)
  local tail="$1"
  ssh "$REMOTE" "for n in ecomesh_${tail} ${tail}; do [ -d ${REMOTE_ROOT}/sats/training/runs/\$n ] && { echo \$n; break; }; done" 2>/dev/null
}

complete() {                          # 완주한 run 만 동기화(학습 중 반쪽 복사 방지)
  local host_dir="$1"                 # "local:<path>" 또는 "remote:<path>"
  local py='import json,sys; h=json.load(open(sys.argv[1])); e=h[-1] if isinstance(h,list) else h; c=json.load(open(sys.argv[2])); sys.exit(0 if e.get("epoch",0)>=c.get("epochs",1) else 1)'
  case "$host_dir" in
    local:*)  d="${host_dir#local:}"
              python3 -c "$py" "$d/history.json" "$d/config.json" 2>/dev/null ;;
    remote:*) d="${host_dir#remote:}"
              ssh "$REMOTE" "python3 -c '$py' '$d/history.json' '$d/config.json'" 2>/dev/null ;;
  esac
}

echo "== SATS 배포 run =="
for v in v4 v5 v6 v7 v8 v9; do
  for t in g05nsc g05ns g025 g01 all4; do
    tail="${v}_deploy_${t}"
    local_dir=""
    for n in "ecomesh_${tail}" "${tail}"; do
      [ -d "$LOCAL_ROOT/sats/training/runs/$n" ] && local_dir="$n" && break
    done
    remote_dir="$(remote_name "$tail")"
    if [ -n "$local_dir" ] && [ -z "$remote_dir" ]; then
      if complete "local:$LOCAL_ROOT/sats/training/runs/$local_dir"; then
        echo "  → push $local_dir"
        rsync -a "$LOCAL_ROOT/sats/training/runs/$local_dir" "$REMOTE:${REMOTE_ROOT}/sats/training/runs/"
      else
        echo "  ~ skip $local_dir (미완주 — 학습 중이거나 중단)"
      fi
    elif [ -z "$local_dir" ] && [ -n "$remote_dir" ]; then
      if complete "remote:sensor_training/sats/training/runs/$remote_dir"; then
        echo "  ← pull $remote_dir"
        rsync -a "$REMOTE:${REMOTE_ROOT}/sats/training/runs/$remote_dir" "$LOCAL_ROOT/sats/training/runs/"
      else
        echo "  ~ skip $remote_dir (미완주 — 학습 중이거나 중단)"
      fi
    fi
  done
done

echo "== v0 클린 베이스 =="
for n in ecomesh_xy0p5_base_nosize; do
  if [ -d "$LOCAL_ROOT/sats/training/runs/$n" ] && [ -f "$LOCAL_ROOT/sats/training/runs/$n/best_model.pt" ]; then
    rsync -a "$LOCAL_ROOT/sats/training/runs/$n" "$REMOTE:${REMOTE_ROOT}/sats/training/runs/" && echo "  → push $n"
  fi
done

echo "== bending 산출물(deform 복원기·estimator) =="
rsync -a --ignore-existing "$REMOTE:${REMOTE_ROOT}/sats/bending/runs/deform_restorer_*" \
      "$LOCAL_ROOT/sats/bending/runs/" 2>/dev/null && echo "  ← pull deform_restorer_*"
rsync -a --ignore-existing "$LOCAL_ROOT/sats/bending/runs/deform_restorer_"* \
      "$REMOTE:${REMOTE_ROOT}/sats/bending/runs/" 2>/dev/null || true
rsync -a --ignore-existing "$REMOTE:${REMOTE_ROOT}/sats/bending/runs/estimator_*" \
      "$LOCAL_ROOT/sats/bending/runs/" 2>/dev/null && echo "  ← pull estimator_*"

echo "완료. 확인:"
echo "  local : $(ls $LOCAL_ROOT/sats/training/runs | grep -c deploy) deploy runs"
ssh "$REMOTE" "echo \"  remote: \$(ls ${REMOTE_ROOT}/sats/training/runs | grep -c deploy) deploy runs\""
