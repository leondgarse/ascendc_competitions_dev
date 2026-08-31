// Compare asdBlas (SIP, plan-based) against ops-blas for the same complex ops.
#include <chrono>
#include <cstdio>
#include <vector>
#include <complex>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "blas_api.h"
using namespace AsdSip;
int main(){
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream s=nullptr; aclrtCreateStream(&s);
    asdBlasHandle h{};
    if(asdBlasCreate(h)!=0){printf("create failed\n");return 1;}
    asdBlasSetStream(h,s);
    if(asdBlasMakeCopyPlan(h)!=0){printf("MakeCopyPlan failed\n");return 1;}
    size_t wsSize=0;
    int wrc=(int)asdBlasGetWorkspaceSize(h,wsSize);
    printf("GetWorkspaceSize rc=%d size=%zu\n",wrc,wsSize);
    void* ws=nullptr;
    if(wsSize>0) aclrtMalloc(&ws,wsSize,ACL_MEM_MALLOC_HUGE_FIRST);
    printf("SetWorkspace rc=%d\n",(int)asdBlasSetWorkspace(h,ws));
    for(int n : {512,4096}){
        size_t bytes=(size_t)n*8;
        void *dX,*dY; aclrtMalloc(&dX,bytes,ACL_MEM_MALLOC_HUGE_FIRST); aclrtMalloc(&dY,bytes,ACL_MEM_MALLOC_HUGE_FIRST);
        int64_t shape[1]={n};
        int64_t strides[1]={1};
        aclTensor* tx=aclCreateTensor(shape,1,ACL_COMPLEX64,strides,0,ACL_FORMAT_ND,shape,1,dX);
        aclTensor* ty=aclCreateTensor(shape,1,ACL_COMPLEX64,strides,0,ACL_FORMAT_ND,shape,1,dY);
        if(!tx||!ty){printf("tensor create failed\n"); return 1;}
        auto rc=asdBlasCcopy(h,n,tx,1,ty,1);
        if(rc!=0){printf("Ccopy n=%d failed rc=%d\n",n,(int)rc); continue;}
        for(int i=0;i<20;++i) asdBlasCcopy(h,n,tx,1,ty,1);
        aclrtSynchronizeStream(s);
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<100;++i) asdBlasCcopy(h,n,tx,1,ty,1);
        aclrtSynchronizeStream(s);
        auto t1=std::chrono::high_resolution_clock::now();
        printf("asdBlasCcopy n=%-6d: %8.2f us\n",n,std::chrono::duration<double,std::micro>(t1-t0).count()/100);
        aclrtFree(dX);aclrtFree(dY);
    }
    aclrtDestroyStream(s);aclrtResetDevice(0);aclFinalize();
    return 0;
}
