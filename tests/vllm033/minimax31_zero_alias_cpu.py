"""CPU-only ownership/layout checks; HIP registration is explicitly mocked."""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.append('/usr/local/lib/python3.12/site-packages')
import torch
from hcu_vllm_das.parallel import minimax31_hisparse as module
from hcu_vllm_das.parallel.minimax31_decode_attention import Verify8Runtime
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks/decode'))
from minimax31_release_history import release


class FakeHIP:
    def __init__(self):self.live={}
    def hipHostRegister(self,pointer,size,flags):
        assert flags==2 and pointer.value not in self.live
        self.live[pointer.value]=size;return 0
    def hipHostUnregister(self,pointer):
        assert pointer.value in self.live
        del self.live[pointer.value];return 0


def main():
    hip=FakeHIP();module.loaded_hip_runtime=lambda:hip
    torch.cuda.is_current_stream_capturing=lambda:False
    shape=(8,4,128);small=4096;capacity=2*60*small
    with tempfile.TemporaryFile() as source:
        source.truncate(capacity)
        backing=module.MappedHostStorage(shape,layer_bytes=small,chunk_bytes=small,
            zero=True,existing_zero_backing_fd=source.fileno(),zero_slice_source_bytes=capacity)
        pointer=backing.storage.data_ptr()
        backing.as_readonly_zero_layer_alias(60)
        assert backing.storage.shape==(2,60,8,4,128)
        assert backing.storage.stride()[:2]==(0,0)
        assert backing.registered_bytes==small and sum(hip.live.values())==small
        assert all(backing.storage[k,l].is_contiguous() and backing.storage[k,l].data_ptr()==pointer
                   for k in range(2) for l in range(60))
        assert torch.equal(backing.storage,torch.zeros((2,60,*shape),dtype=torch.uint8))
        try:Verify8Runtime.commit(type('State',(),{'backing':backing})(),None)
        except RuntimeError as error:assert 'isolated VERIFY' in str(error)
        else:raise AssertionError('readonly alias commit accepted')
        for kwargs in [dict(zero=False,zero_slice_source_bytes=capacity),
                       dict(zero=True,zero_slice_source_bytes=capacity+4096),
                       dict(zero=True,zero_slice_source_bytes=small//2)]:
            try:module.MappedHostStorage(shape,layer_bytes=small,chunk_bytes=small,
                existing_zero_backing_fd=source.fileno(),**kwargs)
            except ValueError:pass
            else:raise AssertionError('invalid zero slice accepted')
        full=module.MappedHostStorage((2,60,*shape),layer_bytes=small,chunk_bytes=small,
            zero=True,existing_zero_backing_fd=source.fileno())
        assert len(hip.live)==121
        calls=[]
        class World:
            def barrier(self):calls.append('barrier')
        torch.cuda.synchronize=lambda:calls.append('gpu_drained')
        release(runtime=SimpleNamespace(backing=full,bootstrap_history_backing=backing),rank=0,world=World())
        assert calls[0]=='gpu_drained' and calls.count('barrier')==9
        assert not hip.live and not backing._finalizer.alive and not full._finalizer.alive
        assert backing.registered_bytes==full.registered_bytes==0
        assert not backing.ranges and not full.ranges
        assert torch.equal(backing.storage,full.storage)
        assert os.fstat(source.fileno()).st_size==capacity
    print('CPU_ZERO_ALIAS_CONTRACT_PASSED: exact full-zero values; one registered slice; contiguous layer views; commit/wrong-capacity rejection; quiescent owned-range release; keeper FD/data retained')


if __name__=='__main__':main()
