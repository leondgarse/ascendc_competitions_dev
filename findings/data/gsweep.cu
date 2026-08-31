#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
int main(int argc,char**argv){
    FILE* f=fopen(argv[1],"r"); if(!f){printf("no input\n");return 1;}
    char line[512]; if(!fgets(line,sizeof line,f)){return 1;}   // header
    cublasHandle_t h; cublasCreate(&h);
    std::mt19937 rng(7); std::normal_distribution<float> nd(0,1);
    printf("id,n,k,uplo,trans,diag,incx,csv_ms,a100_us\n");
    while(fgets(line,sizeof line,f)){
        char id[64],up[16],tr[16],dg[16]; int n,k,incx; double csv;
        if(sscanf(line,"%63[^,],%d,%d,%15[^,],%15[^,],%15[^,],%d,%lf",id,&n,&k,up,tr,dg,&incx,&csv)!=8) continue;
        auto U = strcmp(up,"UPPER")==0?CUBLAS_FILL_MODE_UPPER:CUBLAS_FILL_MODE_LOWER;
        auto T = strcmp(tr,"N")==0?CUBLAS_OP_N:(strcmp(tr,"T")==0?CUBLAS_OP_T:CUBLAS_OP_C);
        auto D = strcmp(dg,"UNIT")==0?CUBLAS_DIAG_UNIT:CUBLAS_DIAG_NON_UNIT;
        int lda=k+1;
        std::vector<cuComplex> A((size_t)lda*n),x((size_t)n);
        for(auto&v:A)v=make_cuComplex(nd(rng),nd(rng));
        for(auto&v:x)v=make_cuComplex(nd(rng),nd(rng));
        cuComplex*dA,*dX;
        if(cudaMalloc(&dA,A.size()*8)!=cudaSuccess){continue;}
        if(cudaMalloc(&dX,x.size()*8)!=cudaSuccess){cudaFree(dA);continue;}
        cudaMemcpy(dA,A.data(),A.size()*8,cudaMemcpyHostToDevice);
        cudaMemcpy(dX,x.data(),x.size()*8,cudaMemcpyHostToDevice);
        auto run=[&]{cublasCtbmv(h,U,T,D,n,k,dA,lda,dX,incx);};
        for(int i=0;i<20;++i)run(); cudaDeviceSynchronize();
        const int R=100;
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<R;++i)run(); cudaDeviceSynchronize();
        auto t1=std::chrono::high_resolution_clock::now();
        double us=std::chrono::duration<double,std::micro>(t1-t0).count()/R;
        printf("%s,%d,%d,%s,%s,%s,%d,%.6f,%.3f\n",id,n,k,up,tr,dg,incx,csv,us);
        fflush(stdout);
        cudaFree(dA);cudaFree(dX);
    }
    fclose(f); cublasDestroy(h); return 0;
}
