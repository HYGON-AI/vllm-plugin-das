import sys
sys.path.insert(0,'/root/hy4-perf-stage4')
from repro import rpc,save,r
save('source-guard-install.json',rpc('stage4_exec',str(r/'install_source_guard.py')))
