// asdBlas (SIP, plan-based) vs ops-blas for the same complex ops on 910B3.
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
    printf("%-28s %10s %10s\n","op","n=512","n=4096");
    // ---- SIP Ccopy ----
    {
        asdBlasHandle h{}; asdBlasCreate(h); asdBlasMakeCopyPlan(h);
        size_t ws=0; asdBlasGetWorkspaceSize(h,ws);
        void* w=nullptr; if(ws) aclrtMalloc(&w,ws,ACL_MEM_MALLOC_HUGE_FIRST);
        asdBlasSetWorkspace(h,w); asdBlasSetStream(h,s);
        double r[2]; int j=0;
        for(int n : {512,4096}){
            void*dx,*dy; aclrtMalloc(&dx,(size_t)n*8,ACL_MEM_MALLOC_HUGE_FIRST); aclrtMalloc(&dy,(size_t)n*8,ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMemset(dx,(size_t)n*8,0,(size_t)n*8);
            auto tx=mk(dx,n,ACL_COMPLEX64), ty=mk(dy,n,ACL_COMPLEX64);
            for(int i=0;i<20;++i) asdBlasCcopy(h,n,tx,1,ty,1);
            aclrtSynchronizeStream(s);
            auto t0=std::chrono::high_resolution_clock::now();
            for(int i=0;i<100;++i) asdBlasCcopy(h,n,tx,1,ty,1);
            aclrtSynchronizeStream(s);
            auto t1=std::chrono::high_resolution_clock::now();
            r[j++]=std::chrono::duration<double,std::micro>(t1-t0).count()/100;
            aclrtFree(dx);aclrtFree(dy);
        }
        printf("%-28s %10.2f %10.2f\n","asdBlasCcopy (SIP)",r[0],r[1]);
        asdBlasDestroy(h);
    }
    // ---- ops-blas ccopy ----
    {
        aclblasHandle_t h=nullptr; aclblasCreate(&h); aclblasSetStream(h,s);
        double r[2]; int j=0;
        for(int n : {512,4096}){
            void*dx,*dy; aclrtMalloc(&dx,(size_t)n*8,ACL_MEM_MALLOC_HUGE_FIRST); aclrtMalloc(&dy,(size_t)n*8,ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMemset(dx,(size_t)n*8,0,(size_t)n*8);
            for(int i=0;i<20;++i) aclblasCcopy(h,n,(aclblasComplex*)dx,1,(aclblasComplex*)dy,1);
            aclrtSynchronizeStream(s);
            auto t0=std::chrono::high_resolution_clock::now();
            for(int i=0;i<100;++i) aclblasCcopy(h,n,(aclblasComplex*)dx,1,(aclblasComplex*)dy,1);
            aclrtSynchronizeStream(s);
            auto t1=std::chrono::high_resolution_clock::now();
            r[j++]=std::chrono::duration<double,std::micro>(t1-t0).count()/100;
            aclrtFree(dx);aclrtFree(dy);
        }
        printf("%-28s %10.2f %10.2f\n","aclblasCcopy (ops-blas)",r[0],r[1]);
        aclblasDestroy(h);
    }
    aclrtDestroyStream(s); aclrtResetDevice(0); aclFinalize();
    return 0;
}
