// Runtime workaround for the exact legacy RPC wheel in this PCP16/DP16 test.
// Default behavior is unchanged. Only explicit target/pcp0 scopes skip module
// initialization on processes that never initiate GPU copies in this topology.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>

extern "C" void _ZN8mooncake12HipTransport14loadCopyModuleEv(void* self) {
    constexpr const char* symbol="_ZN8mooncake12HipTransport14loadCopyModuleEv";
    void* native=dlopen("/usr/local/lib/python3.10/dist-packages/mooncake/libtransfer_engine.so",RTLD_NOW|RTLD_NOLOAD);
    if (!native) {std::fprintf(stderr,"RPC_MODULE_SCOPE native library missing: %s\n",dlerror());std::abort();}
    auto original=reinterpret_cast<void(*)(void*)>(dlsym(native,symbol));
    auto get_device=reinterpret_cast<int(*)(int*)>(dlsym(native,"hipGetDevice"));
    if (!original || !get_device || original==&_ZN8mooncake12HipTransport14loadCopyModuleEv) {
        std::fprintf(stderr,"RPC_MODULE_SCOPE symbol resolution failed\n");std::abort();
    }
    int device=-1;
    if (get_device(&device)!=0) {std::fprintf(stderr,"RPC_MODULE_SCOPE hipGetDevice failed\n");std::abort();}
    const char* scope=std::getenv("MC_RPC_COPY_MODULE_SCOPE");
    const bool target=scope && std::strcmp(scope,"target")==0;
    const bool pcp0=scope && std::strcmp(scope,"pcp0")==0;
    if (scope && !target && !pcp0 && std::strcmp(scope,"full")!=0) {
        std::fprintf(stderr,"RPC_MODULE_SCOPE invalid scope: %s\n",scope);std::abort();
    }
    const bool skip=target || (pcp0 && device!=0);
    std::fprintf(stderr,"RPC_MODULE_SCOPE pid=%ld device=%d scope=%s action=%s\n",static_cast<long>(getpid()),device,scope?scope:"full",skip?"skip_unused_module":"original_module_load");
    if (!skip) original(self);
    dlclose(native);
}
