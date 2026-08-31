// What is the irreducible cost of a <<<>>> kernel launch on 910B3?
// Compare: our ctbmv (k=0), ops-blas ccopy, SIP Ccopy, and a raw async memcpy.
#include <chrono>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "blas_api.h"
#include "cann_ops_blas.h"
using namespace AsdSip;
static aclTensor* mk(void* d,int n,aclDataType dt){
    int64_t shp[1]={n}, str[1]={1};
    return aclCreateTensor(shp,1,dt,str,0,ACL_FORMAT_ND,shp,1,d);
}
int main(){
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream s=nullptr; aclrtCreateStream(&s);
    const int n=512; const size_t B=(size_t)n*8;
    void *dx,*dy,*dA;
    aclrtMalloc(&dx,B,ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dy,B,ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dA,B*2,ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dx,B,0,B); aclrtMemset(dA,B*2,0,B*2);

    auto bench=[&](const char*nm, auto f){
        for(int i=0;i<30;++i) f(); aclrtSynchronizeStream(s);
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<200;++i) f();
        aclrtSynchronizeStream(s);
        auto t1=std::chrono::high_resolution_clock::now();
        printf("  %-34s %8.2f us\n",nm,std::chrono::duration<double,std::micro>(t1-t0).count()/200);
    };

    printf("n=%d, complex64 (%zu bytes)\n",n,B);
    bench("aclrtMemcpyAsync D2D (no kernel)", [&]{ aclrtMemcpyAsync(dy,B,dx,B,ACL_MEMCPY_DEVICE_TO_DEVICE,s); });

    { aclblasHandle_t h=nullptr; aclblasCreate(&h); aclblasSetStream(h,s);
      bench("aclblasCcopy (ops-blas)", [&]{ aclblasCcopy(h,n,(aclblasComplex*)dx,1,(aclblasComplex*)dy,1); });
      bench("aclblasCtbmv k=0 (ours)", [&]{ aclblasCtbmv(h,ACLBLAS_UPPER,ACLBLAS_OP_N,ACLBLAS_NON_UNIT,n,0,(const aclblasComplex*)dA,1,(aclblasComplex*)dx,1); });
      bench("aclblasCtbmv k=8 (ours)", [&]{ aclblasCtbmv(h,ACLBLAS_UPPER,ACLBLAS_OP_N,ACLBLAS_NON_UNIT,n,8,(const aclblasComplex*)dA,9,(aclblasComplex*)dx,1); });
      aclblasDestroy(h); }

    { asdBlasHandle h{}; asdBlasCreate(h); asdBlasMakeCopyPlan(h);
      size_t ws=0; asdBlasGetWorkspaceSize(h,ws);
      void* w=nullptr; if(ws) aclrtMalloc(&w,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      asdBlasSetWorkspace(h,w); asdBlasSetStream(h,s);
      auto tx=mk(dx,n,ACL_COMPLEX64), ty=mk(dy,n,ACL_COMPLEX64);
      bench("asdBlasCcopy (SIP plan)", [&]{ asdBlasCcopy(h,n,tx,1,ty,1); });
      asdBlasDestroy(h); }

    aclrtDestroyStream(s); aclrtResetDevice(0); aclFinalize();
    return 0;
}
