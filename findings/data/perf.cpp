#include <chrono>
#include <cstdio>
#include <complex>
#include <vector>
#include <random>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
using cplx=std::complex<float>;
struct C{int n,k;const char*uplo;const char*trans;const char*diag;int incx;double target;};
int main(){
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream s=nullptr; aclrtCreateStream(&s);
    aclblasHandle_t h=nullptr; aclblasCreate(&h); aclblasSetStream(h,s);
    C cases[]={
        {512,8,"U","N","NON_UNIT",1,10.0},
        {1024,16,"U","N","NON_UNIT",1,16.51},
        {2048,32,"L","T","NON_UNIT",1,19.77},
    };
    std::mt19937 rng(1); std::normal_distribution<float> nd(0,1);
    printf("%-6s %-5s %-6s %-7s %10s %10s %8s\n","n","k","uplo","trans","avg_us","target_us","verdict");
    for(auto&c:cases){
        int lda=c.k+1;
        std::vector<cplx> A((size_t)lda*c.n), x(c.n);
        for(auto&v:A) v=cplx(nd(rng),nd(rng));
        for(auto&v:x) v=cplx(nd(rng),nd(rng));
        void*dA,*dX; size_t aB=A.size()*sizeof(cplx), xB=x.size()*sizeof(cplx);
        aclrtMalloc(&dA,aB,ACL_MEM_MALLOC_HUGE_FIRST); aclrtMalloc(&dX,xB,ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dA,aB,A.data(),aB,ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dX,xB,x.data(),xB,ACL_MEMCPY_HOST_TO_DEVICE);
        auto UP=c.uplo[0]=='U'?ACLBLAS_UPPER:ACLBLAS_LOWER;
        auto TR=c.trans[0]=='N'?ACLBLAS_OP_N:(c.trans[0]=='T'?ACLBLAS_OP_T:ACLBLAS_OP_C);
        auto run=[&]{ aclblasCtbmv(h,UP,TR,ACLBLAS_NON_UNIT,c.n,c.k,(const aclblasComplex*)dA,lda,(aclblasComplex*)dX,c.incx); };
        for(int i=0;i<20;++i) run();                       // warmup
        aclrtSynchronizeStream(s);
        const int R=100;
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<R;++i) run();
        aclrtSynchronizeStream(s);
        auto t1=std::chrono::high_resolution_clock::now();
        double us=std::chrono::duration<double,std::micro>(t1-t0).count()/R;
        printf("%-6d %-5d %-6s %-7s %10.2f %10.2f %8s\n",c.n,c.k,c.uplo,c.trans,us,c.target, us<=c.target?"PASS":"SLOW");
        aclrtFree(dA);aclrtFree(dX);
    }
    aclblasDestroy(h);aclrtDestroyStream(s);aclrtResetDevice(0);aclFinalize();
    return 0;
}
