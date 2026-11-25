// Execution-only raw allocator for a Torch MemPool; the global allocator stays native.
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <sys/types.h>
#include <unordered_map>

struct ihipStream_t;
using hipStream_t=ihipStream_t*;
extern "C" int hipGetDevice(int*);
extern "C" int hipSetDevice(int);
extern "C" int hipMallocAsync(void**,size_t,hipStream_t);
extern "C" int hipFreeAsync(void*,hipStream_t);

namespace {
struct Allocation { size_t bytes; int device; };
// Retain CPU bookkeeping until process exit, including late pool destruction.
auto* allocations=new std::unordered_map<void*,Allocation>;
auto* mutex=new std::mutex;
std::atomic<unsigned long long> calls{0},frees{0},errors{0},total_bytes{0};
std::atomic<int> expected_device{-1};
void failed(const char* operation,int status,size_t bytes,int device) {
  ++errors;
  std::fprintf(stderr,"MINIMAX_EXECUTION_POOL_ERROR op=%s status=%d bytes=%zu device=%d\n",
               operation,status,bytes,device);
}
struct DeviceScope {
  int previous=-1; bool valid=false;
  explicit DeviceScope(int device) {
    int status=hipGetDevice(&previous);
    if (!status && previous!=device) status=hipSetDevice(device);
    valid=!status;
    if(status) failed("device_guard",status,0,device);
  }
  ~DeviceScope() { if(valid && previous>=0) hipSetDevice(previous); }
};
}

extern "C" int m31_pool_set_device(int device) {
  int previous=-1;
  return expected_device.compare_exchange_strong(previous,device) || previous==device ? 0 : 1;
}
extern "C" void* m31_pool_malloc(ssize_t bytes,int device,hipStream_t stream) {
  if(bytes<0 || device!=expected_device.load()) {
    failed("allocation_device",-1,bytes<0?0:size_t(bytes),device);return nullptr;
  }
  DeviceScope scope(device);if(!scope.valid)return nullptr;
  void* pointer=nullptr;int status=hipMallocAsync(&pointer,size_t(bytes),stream);
  if(status || (!pointer && bytes)) {failed("hipMallocAsync",status,size_t(bytes),device);return nullptr;}
  if(pointer) {
    std::lock_guard<std::mutex> guard(*mutex);
    if(!allocations->emplace(pointer,Allocation{size_t(bytes),device}).second) {
      failed("duplicate_live_pointer",-1,size_t(bytes),device);return nullptr;
    }
  }
  ++calls;total_bytes+=size_t(bytes);return pointer;
}
extern "C" void m31_pool_free(void* pointer,size_t bytes,int device,hipStream_t stream) {
  if(!pointer)return;
  {
    std::lock_guard<std::mutex> guard(*mutex);
    auto entry=allocations->find(pointer);
    if(entry==allocations->end() || entry->second.bytes!=bytes || entry->second.device!=device) {
      failed("foreign_or_mismatched_free",-1,bytes,device);return;
    }
  }
  DeviceScope scope(device);if(!scope.valid)return;
  int status=hipFreeAsync(pointer,stream);
  if(status) {failed("hipFreeAsync",status,bytes,device);return;}
  {std::lock_guard<std::mutex> guard(*mutex);allocations->erase(pointer);}
  ++frees;
}
extern "C" unsigned long long m31_pool_allocations(){return calls.load();}
extern "C" unsigned long long m31_pool_frees(){return frees.load();}
extern "C" unsigned long long m31_pool_errors(){return errors.load();}
extern "C" unsigned long long m31_pool_total_requested_bytes(){return total_bytes.load();}
