#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <complex>
#include <random>
#include "acl/acl.h"
#include "cann_ops_blas.h"
using cplx=std::complex<float>;
int main(int argc,char**argv){
    FILE* f=fopen(argv[1],"r"); if(!f) return 1;
    char line[512]; if(!fgets(line,sizeof line,f)) return 1;
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream s=nullptr; aclrtCreateStream(&s);
    aclblasHandle_t h=nullptr; aclblasCreate(&h); aclblasSetStream(h,s);
    std::mt19937 rng(7); std::normal_distribution<float> nd(0,1);
    printf("id,n,k,uplo,trans,diag,incx,a100_us,npu_us\n");
    while(fgets(line,sizeof line,f)){
        char id[64],up[16],tr[16],dg[16]; int n,k,incx; double a100;
        if(sscanf(line,"%63[^,],%d,%d,%15[^,],%15[^,],%15[^,],%d,%lf",id,&n,&k,up,tr,dg,&incx,&a100)!=8) continue;
        auto U=strcmp(up,"UPPER")==0?ACLBLAS_UPPER:ACLBLAS_LOWER;
        auto T=strcmp(tr,"N")==0?ACLBLAS_OP_N:(strcmp(tr,"T")==0?ACLBLAS_OP_T:ACLBLAS_OP_C);
        auto D=strcmp(dg,"UNIT")==0?ACLBLAS_UNIT:ACLBLAS_NON_UNIT;
        int lda=k+1;
        std::vector<cplx> A((size_t)lda*n),x((size_t)n);
        for(auto&v:A)v=cplx(nd(rng),nd(rng)); for(auto&v:x)v=cplx(nd(rng),nd(rng));
        void*dA,*dX; size_t aB=A.size()*8,xB=x.size()*8;
        if(aclrtMalloc(&dA,aB,ACL_MEM_MALLOC_HUGE_FIRST)!=ACL_SUCCESS) continue;
        if(aclrtMalloc(&dX,xB,ACL_MEM_MALLOC_HUGE_FIRST)!=ACL_SUCCESS){aclrtFree(dA);continue;}
        aclrtMemcpy(dA,aB,A.data(),aB,ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dX,xB,x.data(),xB,ACL_MEMCPY_HOST_TO_DEVICE);
        auto run=[&]{aclblasCtbmv(h,U,T,D,n,k,(const aclblasComplex*)dA,lda,(aclblasComplex*)dX,incx);};
        for(int i=0;i<20;++i)run(); aclrtSynchronizeStream(s);
        const int R=60;
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<R;++i)run(); aclrtSynchronizeStream(s);
        auto t1=std::chrono::high_resolution_clock::now();
        printf("%s,%d,%d,%s,%s,%s,%d,%.3f,%.3f\n",id,n,k,up,tr,dg,incx,a100,
               std::chrono::duration<double,std::micro>(t1-t0).count()/R);
        fflush(stdout);
        aclrtFree(dA);aclrtFree(dX);
    }
    aclblasDestroy(h);aclrtDestroyStream(s);aclrtResetDevice(0);aclFinalize();
    return 0;
}
