"""Validate the fused column-tile scheme before writing kernel code.

Scheme: process output rows in tiles. For a tile [r0, r0+R):
  - the needed banded columns are a contiguous range [c0, c1)
  - load them as (k+1) x C block (each column contiguous -> real DMA burst)
  - for each band d in 0..k, the contribution is a fixed ROW of that block
  - accumulate into y[r0..r0+R) entirely in UB
"""
import numpy as np, itertools

def dense(A,n,k,uplo):
    M=np.zeros((n,n),dtype=complex)
    for j in range(n):
        if uplo=='U':
            for i in range(max(0,j-k),j+1): M[i,j]=A[k+i-j,j]
        else:
            for i in range(j,min(n,j+k+1)): M[i,j]=A[i-j,j]
    return M

def ref(A,x,n,k,uplo,trans,diag):
    M=dense(A,n,k,uplo)
    if diag=='U': np.fill_diagonal(M,1+0j)
    op = M if trans=='N' else (M.T if trans=='T' else M.conj().T)
    return op@x

def fused(A,x,n,k,uplo,trans,diag,R):
    """R = row-tile height."""
    y=np.zeros(n,dtype=complex)
    isT = trans in ('T','C')
    for r0 in range(0,n,R):
        rows=min(R,n-r0)
        acc=np.zeros(rows,dtype=complex)
        for d in range(k+1):
            # band d: which output rows in this tile get a contribution, and from which j
            # Reuse the validated band mapping:
            #   L,N : y[col+d] += A[d, col]     * x[col]
            #   L,T : y[col-d] += A[d, col-d]   * x[col]
            #   U,N : y[col-d] += A[k-d, col]   * x[col]
            #   U,T : y[col+d] += A[k-d, col+d] * x[col]
            if uplo=='L' and not isT: aRow, yshift = d, +d
            elif uplo=='L' and isT:   aRow, yshift = d, -d
            elif uplo=='U' and not isT: aRow, yshift = k-d, -d
            else:                       aRow, yshift = k-d, +d
            for t in range(rows):
                i = r0+t                      # output row
                col = i - yshift              # source column index
                if col < 0 or col >= n: continue
                if uplo=='L' and isT:   aCol = col-d
                elif uplo=='U' and isT: aCol = col+d
                else:                   aCol = col
                if aCol < 0 or aCol >= n: continue
                a = 1+0j if (diag=='U' and d==0) else A[aRow,aCol]
                if trans=='C' and not (diag=='U' and d==0): a = np.conj(a)
                acc[t] += a*x[col]
        y[r0:r0+rows]=acc
    return y

rng=np.random.default_rng(11); bad=0; tot=0
for n,k in [(8,2),(1,0),(5,0),(7,6),(16,3),(9,8),(6,1),(33,5),(64,8)]:
    for uplo,trans,diag in itertools.product('UL','NTC','NU'):
        for R in (4,8,1000):
            A=rng.normal(size=(k+1,n))+1j*rng.normal(size=(k+1,n))
            x=rng.normal(size=n)+1j*rng.normal(size=n)
            r=ref(A,x,n,k,uplo,trans,diag); g=fused(A,x,n,k,uplo,trans,diag,R); tot+=1
            if not np.allclose(r,g,atol=1e-10):
                bad+=1
                if bad<=6: print(f"  MISMATCH n={n} k={k} {uplo}{trans}{diag} R={R} err={np.abs(r-g).max():.2e}")
print(f"fused scheme: {tot} configs, {bad} mismatches")
