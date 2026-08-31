#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <vector>
#include <random>
#include "acl/acl.h"
#include "cann_ops_blas.h"

using cplx = std::complex<float>;

// Dense golden from banded storage (Netlib semantics).
static void golden(const std::vector<cplx>& A, std::vector<cplx>& x, int n, int k, int lda,
                   char uplo, char trans, char diag) {
    std::vector<cplx> M(static_cast<size_t>(n) * n, cplx(0.f, 0.f));
    for (int j = 0; j < n; ++j) {
        if (uplo == 'U') {
            for (int i = std::max(0, j - k); i <= j; ++i) M[static_cast<size_t>(i) * n + j] = A[static_cast<size_t>(k + i - j) + static_cast<size_t>(j) * lda];
        } else {
            for (int i = j; i < std::min(n, j + k + 1); ++i) M[static_cast<size_t>(i) * n + j] = A[static_cast<size_t>(i - j) + static_cast<size_t>(j) * lda];
        }
    }
    if (diag == 'U') for (int i = 0; i < n; ++i) M[static_cast<size_t>(i) * n + i] = cplx(1.f, 0.f);
    std::vector<cplx> y(n, cplx(0.f, 0.f));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            cplx a = (trans == 'N') ? M[static_cast<size_t>(i) * n + j]
                   : (trans == 'T') ? M[static_cast<size_t>(j) * n + i]
                                    : std::conj(M[static_cast<size_t>(j) * n + i]);
            y[i] += a * x[j];
        }
    x = y;
}

int main() {
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream stream = nullptr; aclrtCreateStream(&stream);
    aclblasHandle_t handle = nullptr;
    if (aclblasCreate(&handle) != ACLBLAS_STATUS_SUCCESS) { printf("create failed\n"); return 1; }
    aclblasSetStream(handle, stream);

    std::mt19937 rng(42); std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0, total = 0;
    const char* U = "UL"; const char* T = "NTC"; const char* D = "NU";
    int shapes[][2] = {{8,2},{16,3},{5,0},{7,6},{1,0},{64,8},{129,5}};

    for (auto& sh : shapes) for (int ui=0;ui<2;++ui) for (int ti=0;ti<3;++ti) for (int di=0;di<2;++di) {
        int n = sh[0], k = sh[1], lda = k + 1;
        char uplo = U[ui], trans = T[ti], diag = D[di];
        std::vector<cplx> A(static_cast<size_t>(lda) * n), x(n), xref;
        for (auto& v : A) v = cplx(nd(rng), nd(rng));
        for (auto& v : x) v = cplx(nd(rng), nd(rng));
        xref = x; golden(A, xref, n, k, lda, uplo, trans, diag);

        void *dA=nullptr,*dX=nullptr;
        size_t aB = A.size()*sizeof(cplx), xB = x.size()*sizeof(cplx);
        aclrtMalloc(&dA,aB,ACL_MEM_MALLOC_HUGE_FIRST); aclrtMalloc(&dX,xB,ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dA,aB,A.data(),aB,ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dX,xB,x.data(),xB,ACL_MEMCPY_HOST_TO_DEVICE);

        auto st = aclblasCtbmv(handle,
            uplo=='U'?ACLBLAS_UPPER:ACLBLAS_LOWER,
            trans=='N'?ACLBLAS_OP_N:(trans=='T'?ACLBLAS_OP_T:ACLBLAS_OP_C),
            diag=='N'?ACLBLAS_NON_UNIT:ACLBLAS_UNIT,
            n,k,reinterpret_cast<const aclblasComplex*>(dA),lda,
            reinterpret_cast<aclblasComplex*>(dX),1);
        aclrtSynchronizeStream(stream);
        std::vector<cplx> out(n);
        aclrtMemcpy(out.data(),xB,dX,xB,ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(dA); aclrtFree(dX);
        ++total;
        if (st != ACLBLAS_STATUS_SUCCESS) { printf("n=%d k=%d %c%c%c STATUS=%d\n",n,k,uplo,trans,diag,(int)st); ++fails; continue; }
        float maxerr = 0.f;
        for (int i=0;i<n;++i) maxerr = std::max(maxerr, std::abs(out[i]-xref[i]));
        if (!(maxerr < 1e-3f)) { printf("n=%-4d k=%-2d %c%c%c  MAXERR=%.3e  FAIL\n",n,k,uplo,trans,diag,maxerr); ++fails; }
    }
    printf("\n=== total=%d  failed=%d ===\n", total, fails);
    aclblasDestroy(handle); aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return fails ? 1 : 0;
}
