// Demonstrate that aclblasStbmv_legacy (arch22) misreads column-major banded
// storage when lda > k+1, by comparing against cblas_stbmv(CblasColMajor,...).
#include <cstdio>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "cann_ops_blas.h"
extern "C" {
  void cblas_stbmv(int order,int uplo,int trans,int diag,int n,int k,
                   const float* a,int lda,float* x,int incx);
}
#define ColMajor 102
#define Upper 121
#define NoTrans 111
#define NonUnit 131
int main(){
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream s=nullptr; aclrtCreateStream(&s);
    aclblasHandle_t h=nullptr; aclblasCreate(&h); aclblasSetStream(h,s);

    const int n=4,k=1;
    for (int lda : {2,4}) {                 // k+1 == 2, then padded
        std::vector<float> A((size_t)lda*n,0.f), x(n), xg(n);
        // Column-major banded (Netlib/cuBLAS): element (row,col) at A[row+col*lda]
        // UPPER, k=1: main diag at row k=1, super-diag at row 0.
        for(int j=0;j<n;++j){
            A[(size_t)1 + (size_t)j*lda] = (float)(j+1);        // diag
            if(j>0) A[(size_t)0 + (size_t)j*lda] = (float)(10+j); // super-diag
        }
        for(int i=0;i<n;++i){ x[i]=1.f; xg[i]=1.f; }
        cblas_stbmv(ColMajor,Upper,NoTrans,NonUnit,n,k,A.data(),lda,xg.data(),1);

        void*dA,*dX,*dY; size_t aB=A.size()*4,xB=n*4;
        aclrtMalloc(&dA,aB,ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dX,xB,ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dY,xB,ACL_MEM_MALLOC_HUGE_FIRST);
        std::vector<float> zero(n,0.f);
        aclrtMemcpy(dA,aB,A.data(),aB,ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dX,xB,x.data(),xB,ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dY,xB,zero.data(),xB,ACL_MEMCPY_HOST_TO_DEVICE);
        aclblasStbmv_legacy(h,ACLBLAS_UPPER,ACLBLAS_OP_N,ACLBLAS_NON_UNIT,
                            (const float*)dA,lda,(const float*)dX,(float*)dY,n,k,1);
        aclrtSynchronizeStream(s);
        std::vector<float> got(n);
        aclrtMemcpy(got.data(),xB,dY,xB,ACL_MEMCPY_DEVICE_TO_HOST);
        float md=0; for(int i=0;i<n;++i) md=std::fmax(md,std::fabs(got[i]-xg[i]));
        printf("lda=%d (k+1=%d)  cblas=[",lda,k+1);
        for(int i=0;i<n;++i)printf("%.0f%s",xg[i],i<n-1?" ":"");
        printf("]  stbmv=[");
        for(int i=0;i<n;++i)printf("%.0f%s",got[i],i<n-1?" ":"");
        printf("]  maxdiff=%.1f %s\n",md, md<1e-4?"OK":"MISMATCH");
        aclrtFree(dA);aclrtFree(dX);aclrtFree(dY);
    }
    aclblasDestroy(h);aclrtDestroyStream(s);aclrtResetDevice(0);aclFinalize();
    return 0;
}
