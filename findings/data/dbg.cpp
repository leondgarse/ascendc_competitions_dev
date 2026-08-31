#include <cstdio>
#include <complex>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
using cplx = std::complex<float>;
int main(){
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream s=nullptr; aclrtCreateStream(&s);
    aclblasHandle_t h=nullptr; aclblasCreate(&h); aclblasSetStream(h,s);

    // n=4,k=1,LOWER,N,NON_UNIT. Banded lower: row0=diag, row1=first subdiag.
    int n=4,k=1,lda=2;
    std::vector<cplx> A(lda*n), x(n);
    // A[r + j*lda]; make values easy to trace: diag j -> (j+1,0); subdiag j -> (10+j,0)
    for(int j=0;j<n;++j){ A[0+j*lda]=cplx(j+1,0); A[1+j*lda]=cplx(10+j,0); }
    for(int j=0;j<n;++j) x[j]=cplx(1,0);
    // dense lower banded: M[j][j]=j+1 ; M[j+1][j]=10+j
    // y[i] = sum_j M[i][j]*x[j] = diag_i*1 + (if i>0) (10+(i-1))*1
    printf("expected: ");
    for(int i=0;i<n;++i){ float v=(i+1); if(i>0) v+=10+(i-1); printf("%.1f ",v);} printf("\n");

    void *dA,*dX; size_t aB=A.size()*sizeof(cplx), xB=x.size()*sizeof(cplx);
    aclrtMalloc(&dA,aB,ACL_MEM_MALLOC_HUGE_FIRST); aclrtMalloc(&dX,xB,ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dA,aB,A.data(),aB,ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dX,xB,x.data(),xB,ACL_MEMCPY_HOST_TO_DEVICE);
    auto st=aclblasCtbmv(h,ACLBLAS_LOWER,ACLBLAS_OP_N,ACLBLAS_NON_UNIT,n,k,
        (const aclblasComplex*)dA,lda,(aclblasComplex*)dX,1);
    aclrtSynchronizeStream(s);
    std::vector<cplx> out(n); aclrtMemcpy(out.data(),xB,dX,xB,ACL_MEMCPY_DEVICE_TO_HOST);
    printf("status=%d\ngot:      ",(int)st);
    for(int i=0;i<n;++i) printf("%.1f%+.1fi ",out[i].real(),out[i].imag()); printf("\n");
    aclrtFree(dA);aclrtFree(dX);aclblasDestroy(h);aclrtDestroyStream(s);aclrtResetDevice(0);aclFinalize();
    return 0;
}
