// Benchmark cublasCtbmv on A100 using the task-doc protocol:
// warmup, then >50 timed iterations, average microseconds per call.
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

struct Case { int n, k; cublasFillMode_t uplo; cublasOperation_t trans; cublasDiagType_t diag; const char* label; double doc; };

int main() {
    cublasHandle_t h; cublasCreate(&h);
    Case cases[] = {
        {512,  8,  CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, "512/8   U N NON_UNIT",  7.996},
        {1024, 16, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, "1024/16 U N NON_UNIT", 13.209},
        {2048, 32, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, "2048/32 L T NON_UNIT", 15.817},
        {4096, 64, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, CUBLAS_DIAG_UNIT,     "4096/64 L C UNIT",     22.003},
    };
    std::mt19937 rng(1); std::normal_distribution<float> nd(0,1);
    printf("%-24s %10s %10s %8s\n", "case", "A100_us", "csv_us", "ratio");
    for (auto& c : cases) {
        int lda = c.k + 1;
        std::vector<cuComplex> A((size_t)lda*c.n), x(c.n);
        for (auto& v : A) v = make_cuComplex(nd(rng), nd(rng));
        for (auto& v : x) v = make_cuComplex(nd(rng), nd(rng));
        cuComplex *dA, *dX;
        cudaMalloc(&dA, A.size()*sizeof(cuComplex)); cudaMalloc(&dX, x.size()*sizeof(cuComplex));
        cudaMemcpy(dA, A.data(), A.size()*sizeof(cuComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(dX, x.data(), x.size()*sizeof(cuComplex), cudaMemcpyHostToDevice);
        auto run = [&]{ cublasCtbmv(h, c.uplo, c.trans, c.diag, c.n, c.k, dA, lda, dX, 1); };
        for (int i = 0; i < 20; ++i) run();
        cudaDeviceSynchronize();
        const int R = 100;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < R; ++i) run();
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double,std::micro>(t1-t0).count()/R;
        printf("%-24s %10.3f %10.3f %8.2f\n", c.label, us, c.doc, us/c.doc);
        cudaFree(dA); cudaFree(dX);
    }
    cublasDestroy(h);
    return 0;
}
