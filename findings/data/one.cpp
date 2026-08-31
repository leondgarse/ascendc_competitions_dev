#include <cstdio>
#include <complex>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
using cplx=std::complex<float>;
int main(int argc,char**argv){
    int n=atoi(argv[1]), k=atoi(argv[2]), lda=atoi(argv[3]), incx=atoi(argv[4]);
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream s=nullptr; aclrtCreateStream(&s);
    aclblasHandle_t h=nullptr; aclblasCreate(&h); aclblasSetStream(h,s);
    size_t xphys=(size_t)(n-1)*abs(incx)+1;
    std::vector<cplx> A((size_t)lda*n, cplx(1,0)), x(xphys, cplx(1,0));
    void *dA,*dX; size_t aB=A.size()*sizeof(cplx), xB=x.size()*sizeof(cplx);
    printf("n=%d k=%d lda=%d incx=%d  aB=%zu xB=%zu\n",n,k,lda,incx,aB,xB);
    auto e1=aclrtMalloc(&dA,aB,ACL_MEM_MALLOC_HUGE_FIRST);
    auto e2=aclrtMalloc(&dX,xB,ACL_MEM_MALLOC_HUGE_FIRST);
    printf("malloc: %d %d\n",(int)e1,(int)e2);
    aclrtMemcpy(dA,aB,A.data(),aB,ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dX,xB,x.data(),xB,ACL_MEMCPY_HOST_TO_DEVICE);
    auto st=aclblasCtbmv(h,ACLBLAS_LOWER,ACLBLAS_OP_C,ACLBLAS_NON_UNIT,n,k,
        (const aclblasComplex*)dA,lda,(aclblasComplex*)dX,incx);
    printf("ctbmv status=%d\n",(int)st);
    auto sy=aclrtSynchronizeStream(s); printf("sync=%d\n",(int)sy);
    aclrtFree(dA);aclrtFree(dX);aclblasDestroy(h);aclrtDestroyStream(s);aclrtResetDevice(0);aclFinalize();
    return 0;
}
