"""Read only the task-owned 4 KiB CPU mailboxes, without a CUDA API call."""
import argparse
import json
import os
from pathlib import Path
import struct


def read(path):
    meta=json.loads(path.read_text());pid=meta['pid'];pointer=meta['host_pointer']
    proc=Path('/proc/%d'%pid)
    def alive():return (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==meta['birth']
    if not alive():raise RuntimeError('mailbox owner birth changed')
    found=False
    for line in (proc/'maps').read_text().splitlines():
        fields=line.split();low,high=[int(value,16) for value in fields[0].split('-')]
        if low==pointer and high-low==meta['bytes']==4096 and fields[1]=='rw-s' and '/dev/zero (deleted)' in line:
            found=True;break
    if not found:raise RuntimeError('mailbox is not the declared separate shared page')
    fd=os.open(str(proc/'mem'),os.O_RDONLY|os.O_CLOEXEC)
    try:
        stable=False
        for _ in range(3):
            before=os.pread(fd,4,pointer);data=os.pread(fd,16,pointer);after=os.pread(fd,4,pointer)
            if before==after==data[:4]:stable=True;break
    finally:os.close(fd)
    if not alive():raise RuntimeError('mailbox owner changed during read')
    code,bad,negative,zero=struct.unpack('<4i',data);forward,stage=divmod(code,1000000)
    result={'rank':meta['rank'],'pid':pid,'code':code,'forward_tag':forward,'stable_read':stable,
            'last_moe_route_flags':{'bad':bad,'negative_ids':negative,'zero_weights':zero}}
    if stage==1:result.update(kind='forward',phase='begin')
    elif stage==999999:result.update(kind='forward',phase='end')
    elif 10000<=stage<12000:
        layer,phase=divmod(stage-10000,32)
        result.update(kind='attention',layer=layer,phase=meta['attention_phases'][phase])
    elif 20000<=stage<22000:
        index,phase=divmod(stage-20000,16)
        result.update(kind='moe',moe_index=index,phase=meta['moe_phases'][phase])
    else:result.update(kind='initialization',phase=str(stage))
    return result


parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path)
args=parser.parse_args();rows=[]
for path in sorted(args.root.glob('rank*.gpu_progress.json')):
    try:rows.append(read(path))
    except (OSError,ValueError,RuntimeError,IndexError) as exc:rows.append({'file':path.name,'error':str(exc)})
value={'mailboxes':len(rows),'ranks':rows};text=json.dumps(value,indent=2)+'\n'
if args.output:args.output.write_text(text)
print(text,end='')
