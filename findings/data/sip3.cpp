#include <chrono>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "blas_api.h"
using namespace AsdSip;
static aclTensor* mk(void* d,int n,aclDataType dt){
    int64_t shp[1]={n}, str[1]={1};
    return aclCreateTensor(shp,1,dt,str,0,ACL_FORMAT_ND,shp,1,d);
}
int main(){
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream s=nullptr; aclrtCreateStream(&s);
    // Scopy (real) first: isolates whether the problem is complex-specific
    {
        asdBlasHandle h{};
        printf("Create rc=%d\n",(int)asdBlasCreate(h));
        printf("MakeCopyPlan rc=%d\n",(int)asdBlasMakeCopyPlan(h));
        size_t ws=0; printf("GetWs rc=%d size=%zu\n",(int)asdBlasGetWorkspaceSize(h,ws),ws);
        void* w=nullptr; if(ws) aclrtMalloc(&w,ws,ACL_MEM_MALLOC_HUGE_FIRST);
        printf("SetWs rc=%d\n",(int)asdBlasSetWorkspace(h,w));
        printf("SetStream rc=%d\n",(int)asdBlasSetStream(h,s));
        int n=512; void*dx,*dy;
        aclrtMalloc(&dx,n*4,ACL_MEM_MALLOC_HUGE_FIRST); aclrtMalloc(&dy,n*4,ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(dx,n*4,0,n*4);
        printf("Scopy rc=%d\n",(int)asdBlasScopy(h,n,mk(dx,n,ACL_FLOAT),1,mk(dy,n,ACL_FLOAT),1));
        printf("asdBlasSynchronize rc=%d\n",(int)asdBlasSynchronize(h));
        aclrtSynchronizeStream(s);
        asdBlasDestroy(h);
    }
    // Ccopy with a fresh handle
    {
        asdBlasHandle h{}; asdBlasCreate(h);
        asdBlasMakeCopyPlan(h);
        size_t ws=0; asdBlasGetWorkspaceSize(h,ws);
        void* w=nullptr; if(ws) aclrtMalloc(&w,ws,ACL_MEM_MALLOC_HUGE_FIRST);
        asdBlasSetWorkspace(h,w);
        asdBlasSetStream(h,s);
        int n=512; void*dx,*dy;
        aclrtMalloc(&dx,n*8,ACL_MEM_MALLOC_HUGE_FIRST); aclrtMalloc(&dy,n*8,ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(dx,n*8,0,n*8);
        printf("Ccopy rc=%d\n",(int)asdBlasCcopy(h,n,mk(dx,n,ACL_COMPLEX64),1,mk(dy,n,ACL_COMPLEX64),1));
        printf("asdBlasSynchronize rc=%d\n",(int)asdBlasSynchronize(h));
        aclrtSynchronizeStream(s);
        asdBlasDestroy(h);
    }
    aclrtDestroyStream(s); aclrtResetDevice(0); aclFinalize();
    return 0;
}
