"""Bound validation temporaries to the decoder-hidden warmup allocation size."""
import torch


def finite(tensor):
    flat=tensor.reshape(-1)
    result=torch.ones((),device=tensor.device,dtype=torch.bool)
    for start in range(0,flat.numel(),128*6144):
        result.logical_and_(torch.isfinite(flat[start:start+128*6144]).all())
    return bool(result)


def open_history_lease(path,rank,shape):
    import json,os
    from pathlib import Path
    path=Path(path);record=json.loads(path.read_text())
    if (record.get('schema')!='minimax31-offline-zero-history-lease-v1' or
            record.get('shape')!=list(shape) or record.get('isolated_verify_only') is not True or
            record.get('zero_history') is not True or record.get('commit_allowed') is not False or
            len(record.get('ranks',[]))!=8 or not record.get('generation')):
        raise ValueError('incompatible explicit zero-history lease')
    proc=Path('/proc/%d/stat'%record['pid'])
    def alive():
        return proc.read_text().rsplit(')',1)[1].split()[19]==record['birth']
    if not alive():raise RuntimeError('history keeper birth changed')
    if sorted(row['rank'] for row in record['ranks'])!=list(range(8)):raise ValueError('history ranks differ')
    entry=next(row for row in record['ranks'] if row['rank']==rank)
    fd=os.open('/proc/%d/fd/%d'%(record['pid'],entry['fd']),os.O_RDWR)
    try:
        stat=os.fstat(fd)
        if ((stat.st_ino,stat.st_dev,stat.st_size)!=(entry['inode'],entry['device'],entry['bytes']) or
                stat.st_size!=146158387200 or not alive() or json.loads(path.read_text())!=record):
            raise RuntimeError('history lease identity changed')
    except BaseException:os.close(fd);raise
    print('MINIMAX_ZERO_HISTORY_REUSE '+json_dump({'rank':rank,'generation':record['generation'],
        'keeper_pid':record['pid'],'inode':stat.st_ino,'bytes':stat.st_size}),flush=True)
    return fd


def json_dump(value):
    import json
    return json.dumps(value)
