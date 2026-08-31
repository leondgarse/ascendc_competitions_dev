#include <chrono>
#include <cstdio>
#include <complex>
#include <vector>
#include <random>
#include "acl/acl.h"
#include "cann_ops_blas.h"
using cplx=std::complex<float>;
int main(int argc,char**argv){
    int n=atoi(argv[1]),k=atoi(argv[2]);
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream s=nullptr; aclrtCreateStream(&s);
    aclblasHandle_t h=nullptr; aclblasCreate(&h); aclblasSetStream(h,s);
    int lda=k+1;
    std::mt19937 rng(1); std::normal_distribution<float> nd(0,1);
    std::vector<cplx> A((size_t)lda*n),x(n);
    for(auto&v:A)v=cplx(nd(rng),nd(rng)); for(auto&v:x)v=cplx(nd(rng),nd(rng));
    void*dA,*dX; size_t aB=A.size()*sizeof(cplx),xB=x.size()*sizeof(cplx);
    aclrtMalloc(&dA,aB,ACL_MEM_MALLOC_HUGE_FIRST); aclrtMalloc(&dX,xB,ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dA,aB,A.data(),aB,ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dX,xB,x.data(),xB,ACL_MEMCPY_HOST_TO_DEVICE);
    auto run=[&]{aclblasCtbmv(h,ACLBLAS_UPPER,ACLBLAS_OP_N,ACLBLAS_NON_UNIT,n,k,(const aclblasComplex*)dA,lda,(aclblasComplex*)dX,1);};
    for(int i=0;i<20;++i)run(); aclrtSynchronizeStream(s);
    // batched: measure N calls back-to-back with ONE sync -> isolates per-call device time
    for(int R : {1,10,100}){
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<R;++i) run();
        aclrtSynchronizeStream(s);
        auto t1=std::chrono::high_resolution_clock::now();
        printf("  R=%3d  total=%9.2f us   per-call=%8.2f us\n",R,
            std::chrono::duration<double,std::micro>(t1-t0).count(),
            std::chrono::duration<double,std::micro>(t1-t0).count()/R);
    }
    // host-only cost: time the launch without sync
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<100;++i) run();
    auto t1=std::chrono::high_resolution_clock::now();
    printf("  host launch only (no sync): %.2f us/call\n",std::chrono::duration<double,std::micro>(t1-t0).count()/100);
    aclrtSynchronizeStream(s);
    aclrtFree(dA);aclrtFree(dX);aclblasDestroy(h);aclrtDestroyStream(s);aclrtResetDevice(0);aclFinalize();
    return 0;
}
