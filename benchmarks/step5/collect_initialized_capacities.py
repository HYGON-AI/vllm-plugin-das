"""Collect current startup KV capacities from all four node logs, without GPU imports."""
import argparse, hashlib, json, re
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['prefill-e22','prefill-e23','decode'], required=True)
    p.add_argument('--logs', nargs=4, type=Path, required=True, help='Current logs in p5c1,p5c2,p5c3,p5c4 order')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    dp = 4 if a.profile == 'decode' else 16
    local = dp // 4
    caps, evidence = {}, []
    pattern = r'EngineCore_DP(\d+)[^\n]*GPU KV cache size:\s*([\d,]+) tokens'
    for node, log in enumerate(a.logs):
        raw = log.read_bytes()
        rows = re.findall(pattern, raw.decode('utf-8', errors='replace'))
        found = {}
        for rank, tokens in rows:
            rank, tokens = int(rank), int(tokens.replace(',',''))
            if rank in found:
                raise ValueError('Repeated startup receipt: use one fresh startup log per node')
            found[rank] = tokens
        if set(found) != set(range(node*local,(node+1)*local)) or min(found.values()) <= 0:
            raise ValueError('Missing or unexpected DP capacities in '+str(log))
        caps.update(found)
        evidence.append(dict(log=str(log.resolve()),sha256=hashlib.sha256(raw).hexdigest(),capacities=found))
    required = 64 * (65536 + (4096 if dp == 4 else 1))
    if min(caps.values()) * dp < required:
        raise ValueError('Actual initialized aggregate KV capacity cannot hold this fixed workload')
    # The unchanged benchmark additionally retains its original 0.9 capacity guard.
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(caps,f,indent=2)
    with a.output.with_suffix('.evidence.json').open('x', encoding='utf-8') as f:
        json.dump(dict(profile=a.profile,logs=evidence,capacities=caps),f,indent=2)
    print(a.output)

if __name__ == '__main__': main()
