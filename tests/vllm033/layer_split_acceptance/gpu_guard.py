"""Explicit eight-card selection, physical mapping and conservative idle checks."""
import json
from pathlib import Path
import re
import subprocess


def inventory(smi='/opt/hyhal/bin/hy-smi'):
    raw = subprocess.check_output([smi, '--showbus', '--json'], text=True, timeout=30)
    buses = json.loads(raw[raw.index('{'):])
    topology = {}
    for path in Path('/sys/class/kfd/kfd/topology/nodes').glob('*/properties'):
        props = dict(line.split(None, 1) for line in path.read_text().splitlines() if ' ' in line)
        if not int(props.get('simd_count', '0')):
            continue
        loc = int(props['location_id'])
        pci = '{:04x}:{:02x}:{:02x}.{:x}'.format(
            int(props.get('domain', '0')), loc >> 8, (loc >> 3) & 31, loc & 7)
        topology[pci] = dict(gpu_id=(path.parent / 'gpu_id').read_text().strip(),
                            uuid='GPU-{:016x}'.format(int(props['unique_id'])))
    result = []
    for key, value in buses.items():
        if re.fullmatch(r'card[0-9]+', key):
            pci = value['PCI Bus'].split()[0].lower()
            result.append(dict(topology[pci], smi_index=int(key[4:]), pci=pci))
    if not result or len({r['uuid'] for r in result}) != len(result):
        raise RuntimeError('GPU identity inventory is missing or ambiguous')
    return sorted(result, key=lambda r: r['smi_index'])


def select(all_cards, ids):
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError('Provide exactly eight distinct hy-smi card numbers')
    mapping = {row['smi_index']: row for row in all_cards}
    return [mapping[index] for index in ids]


def snapshot(selected, all_cards):
    devices = {p.resolve().name.lower(): p for p in Path('/sys/class/drm').glob('card[0-9]*/device')
               if (p / 'mem_info_vram_used').is_file()}
    cards = []
    for gpu in selected:
        path = devices[gpu['pci']]
        cards.append(dict(gpu, used_bytes=int((path / 'mem_info_vram_used').read_text()),
                          total_bytes=int((path / 'mem_info_vram_total').read_text()),
                          busy_percent=int((path / 'gpu_busy_percent').read_text())))
    proc_root = Path('/sys/class/kfd/kfd/proc')
    if not proc_root.is_dir():
        raise RuntimeError('KFD process visibility is required')
    clients = []
    for client in proc_root.glob('[0-9]*'):
        try:
            vram = {gpu['pci']: int(path.read_text()) for gpu in all_cards
                    if (path := client / ('vram_' + gpu['gpu_id'])).is_file()}
            allocations = [pci for pci, used in vram.items() if used > 0]
            try:
                executable = str((Path('/proc') / client.name / 'exe').resolve(strict=True))
            except OSError:
                executable = 'unknown'
            if not allocations and executable == '/opt/monitor/v5_exporter/v5_exporter':
                continue
            if not allocations or set(allocations) & {g['pci'] for g in selected}:
                try:
                    context_only = (len(vram) == len(all_cards) and all(used == 0 for used in vram.values())
                                    and not any((client / 'queues').iterdir()))
                except OSError:
                    context_only = False
                clients.append(dict(pid=int(client.name), allocated_pci=allocations,
                                    executable=executable, context_only=context_only))
        except FileNotFoundError:
            continue  # Client exited while the snapshot was taken.
    return dict(cards=cards, clients=clients)


def require_idle(state, idle_mib=2, *, context_idle_mib=None):
    clients = state['clients']
    if clients and (context_idle_mib is None or not all(
            client.get('context_only') and not client.get('allocated_pci') for client in clients)):
        raise RuntimeError('Selected cards have active or unverified KFD clients: ' + json.dumps(state))
    if clients:
        idle_mib = context_idle_mib
    if (len(state['cards']) != 8 or any(
            row['used_bytes'] > idle_mib * 2**20 or row['busy_percent'] != 0
            or row['total_bytes'] <= 0 for row in state['cards'])):
        raise RuntimeError('Selected cards are occupied or process attribution is unknown: ' + json.dumps(state))
    return state
