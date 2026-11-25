// Bounded execution-only MemPool over an independently held native tensor.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <sys/types.h>
#include <unordered_map>
struct ihipStream_t;using hipStream_t=ihipStream_t*;
namespace {
struct Allocation {size_t bytes;int device;};
auto* live=new std::unordered_map<void*,Allocation>;
auto* guard=new std::mutex;
uintptr_t base=0;size_t capacity=0,used=0;
std::atomic<int> expected_device{-1};
std::atomic<unsigned long long> calls{0},frees{0},errors{0},total_bytes{0};
void failure(const char* operation,size_t bytes,int device) {
 ++errors;std::fprintf(stderr,"MINIMAX_EXECUTION_ARENA_ERROR op=%s bytes=%zu device=%d\n",operation,bytes,device);
}
}
extern "C" int m31_pool_set_device(int device) {
 int previous=-1;return expected_device.compare_exchange_strong(previous,device)||previous==device?0:1;
}
extern "C" int m31_pool_set_arena(void* pointer,size_t bytes,int device) {
 std::lock_guard<std::mutex> lock(*guard);
 if(base||!pointer||!bytes||device!=expected_device.load()||uintptr_t(pointer)%256)return 1;
 base=uintptr_t(pointer);capacity=bytes;return 0;
}
extern "C" void* m31_pool_malloc(ssize_t bytes,int device,hipStream_t) {
 std::lock_guard<std::mutex> lock(*guard);
 if(!base||bytes<=0||device!=expected_device.load()){failure("scope",bytes>0?bytes:0,device);return nullptr;}
 size_t aligned=(size_t(bytes)+255)&~size_t(255);
 if(aligned<size_t(bytes)||used>capacity||aligned>capacity-used){failure("capacity",size_t(bytes),device);return nullptr;}
 void* pointer=reinterpret_cast<void*>(base+used);used+=aligned;
 live->emplace(pointer,Allocation{size_t(bytes),device});++calls;total_bytes+=size_t(bytes);return pointer;
}
extern "C" void m31_pool_free(void* pointer,size_t bytes,int device,hipStream_t) {
 if(!pointer)return;
 std::lock_guard<std::mutex> lock(*guard);auto entry=live->find(pointer);
 if(entry==live->end()||entry->second.bytes!=bytes||entry->second.device!=device){failure("foreign_or_mismatched_free",bytes,device);return;}
 // Segments stay backed by the Python-owned native reserve until process exit.
 live->erase(entry);++frees;
}
extern "C" unsigned long long m31_pool_allocations(){return calls.load();}
extern "C" unsigned long long m31_pool_frees(){return frees.load();}
extern "C" unsigned long long m31_pool_errors(){return errors.load();}
extern "C" unsigned long long m31_pool_total_requested_bytes(){return total_bytes.load();}
extern "C" unsigned long long m31_pool_arena_capacity(){return capacity;}
extern "C" unsigned long long m31_pool_arena_used(){return used;}
