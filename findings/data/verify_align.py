"""Validate the alignment-safe variant: every vector op writes at offset 0.

Per band d, process the FULL tile width. Source column for output row i is
i - yshift; when that falls outside [0, n) the lane must contribute zero.
We emulate that by loading x from a clamped base and zeroing the invalid prefix
(yForward) or suffix (!yForward) -- both are contiguous runs at a tile edge, so
Duplicate at offset 0 / a shortened length covers them.
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
    op=M if trans=='N' else (M.T if trans=='T' else M.conj().T)
    return op@x

def kernel(A,x,n,k,uplo,trans,diag,R):
    y=np.zeros(n,dtype=complex)
    isT=trans in ('T','C')
    yForward=(not isT) if uplo=='L' else isT
    for r0 in range(0,n,R):
        rows=min(R,n-r0)
        accR=np.zeros(rows); accI=np.zeros(rows)
        for d in range(k+1):
            # full-width source vectors, zero where invalid  (dst offset always 0)
            xr=np.zeros(rows); xi=np.zeros(rows)
            ar=np.zeros(rows); ai=np.zeros(rows)
            for t in range(rows):
                i=r0+t
                col = i-d if yForward else i+d
                if col<0 or col>=n: continue
                if diag=='U' and d==0:
                    xr[t]=x[col].real; xi[t]=x[col].imag; ar[t]=1.0; ai[t]=0.0
                    continue
                aRow=(k-d) if uplo=='U' else d
                aCol=col
                if isT: aCol = (col+d) if uplo=='U' else (col-d)
                if aCol<0 or aCol>=n: continue
                a=A[aRow,aCol]
                if trans=='C': a=np.conj(a)
                ar[t]=a.real; ai[t]=a.imag
                xr[t]=x[col].real; xi[t]=x[col].imag
            # every op at offset 0, full width
            accR += ar*xr - ai*xi
            accI += ar*xi + ai*xr
        y[r0:r0+rows]=accR+1j*accI
    return y

rng=np.random.default_rng(17); bad=0; tot=0; msgs=[]
for n,k in [(8,2),(1,0),(5,0),(7,6),(16,3),(9,8),(6,1),(33,5),(64,8),(129,5),(512,8)]:
    for uplo,trans,diag in itertools.product('UL','NTC','NU'):
        for R in (4,8,64,10000):
            A=rng.normal(size=(k+1,n))+1j*rng.normal(size=(k+1,n))
            x=rng.normal(size=n)+1j*rng.normal(size=n)
            r=ref(A,x,n,k,uplo,trans,diag); g=kernel(A,x,n,k,uplo,trans,diag,R); tot+=1
            if not np.allclose(r,g,atol=1e-9):
                bad+=1
                if len(msgs)<5: msgs.append(f"n={n} k={k} {uplo}{trans}{diag} R={R} err={np.abs(r-g).max():.2e}")
print(f"alignment-safe scheme: {tot} configs, {bad} mismatches")
for m in msgs: print("  ",m)
