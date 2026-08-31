// Microbenchmark the fixed costs so we can attribute the 61us.
#include <chrono>
#include <cstdio>
#include <vector>
#include <complex>
#include "acl/acl.h"
#include "cann_ops_blas.h"
using cplx=std::complex<float>;
static double bench(aclrtStream s, int R, void(*f)(void*), void* ctx){
    for(int i=0;i<20;++i) f(ctx); aclrtSynchronizeStream(s);
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<R;++i) f(ctx);
    aclrtSynchronizeStream(s);
    auto t1=std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double,std::micro>(t1-t0).count()/R;
}
int main(){
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream s=nullptr; aclrtCreateStream(&s);
    aclblasHandle_t h=nullptr; aclblasCreate(&h); aclblasSetStream(h,s);
    printf("--- pure runtime costs (no kernel) ---\n");
    // 1. empty stream sync
    { auto t0=std::chrono::high_resolution_clock::now();
      for(int i=0;i<1000;++i) aclrtSynchronizeStream(s);
      auto t1=std::chrono::high_resolution_clock::now();
      printf("aclrtSynchronizeStream (already idle): %.3f us\n",
        std::chrono::duration<double,std::micro>(t1-t0).count()/1000); }
    // 2. tiny D2D memcpy async (proxy for one DMA dispatch)
    void *a,*b; aclrtMalloc(&a,4096,ACL_MEM_MALLOC_HUGE_FIRST); aclrtMalloc(&b,4096,ACL_MEM_MALLOC_HUGE_FIRST);
    for(int R:{1,10,100}){
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<R;++i) aclrtMemcpyAsync(b,4096,a,4096,ACL_MEMCPY_DEVICE_TO_DEVICE,s);
        aclrtSynchronizeStream(s);
        auto t1=std::chrono::high_resolution_clock::now();
        printf("memcpyAsync 4KB x%-4d: %8.3f us/op\n",R,
          std::chrono::duration<double,std::micro>(t1-t0).count()/R);
    }
    printf("--- ctbmv scaling with n at k=0 (single kernel path) ---\n");
    for(int n : {1,8,64,512,4096,32768}){
        int k=0,lda=1;
        std::vector<cplx> A((size_t)lda*n,cplx(1,0)), x(n,cplx(1,0));
        void*dA,*dX; aclrtMalloc(&dA,A.size()*8,ACL_MEM_MALLOC_HUGE_FIRST); aclrtMalloc(&dX,x.size()*8,ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dA,A.size()*8,A.data(),A.size()*8,ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dX,x.size()*8,x.data(),x.size()*8,ACL_MEMCPY_HOST_TO_DEVICE);
        for(int i=0;i<20;++i) aclblasCtbmv(h,ACLBLAS_UPPER,ACLBLAS_OP_N,ACLBLAS_NON_UNIT,n,k,(const aclblasComplex*)dA,lda,(aclblasComplex*)dX,1);
        aclrtSynchronizeStream(s);
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<100;++i) aclblasCtbmv(h,ACLBLAS_UPPER,ACLBLAS_OP_N,ACLBLAS_NON_UNIT,n,k,(const aclblasComplex*)dA,lda,(aclblasComplex*)dX,1);
        aclrtSynchronizeStream(s);
        auto t1=std::chrono::high_resolution_clock::now();
        printf("  n=%-6d k=0: %8.2f us   (%.1f KB traffic)\n",n,
          std::chrono::duration<double,std::micro>(t1-t0).count()/100, n*8*3/1024.0);
        aclrtFree(dA);aclrtFree(dX);
    }
    aclblasDestroy(h);aclrtDestroyStream(s);aclrtResetDevice(0);aclFinalize();
    return 0;
}
