"""Model the band-SLICED variant exactly as the kernel does it, to find the bug."""
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

def kernel(A,x,n,k,uplo,trans,diag,R,SLICE):
    y=np.zeros(n,dtype=complex)
    isT=trans in ('T','C')
    yForward = (not isT) if uplo=='L' else isT
    for r0 in range(0,n,R):
        rows=min(R,n-r0); acc=np.zeros(rows,dtype=complex)
        for d0 in range(0,k+1,SLICE):
            dEnd=min(d0+SLICE,k+1); sliceRows=dEnd-d0
            colLo,colHi=r0,r0+rows
            if yForward: colLo = max(0, r0-(dEnd-1))
            else:        colHi = r0+rows+(dEnd-1)
            colHi=min(colHi,n)
            if colLo>=colHi: continue
            blockRowLo = (k-(dEnd-1)) if uplo=='U' else d0
            # staged block: columns [colLo,colHi), rows [blockRowLo, blockRowLo+sliceRows)
            for d in range(d0,dEnd):
                aRow = (k-d) if uplo=='U' else d
                lo,hi=r0,r0+rows
                if yForward:
                    lo=max(lo,d)
                else:
                    if n<d: break
                    hi=min(hi,n-d)
                if lo>=hi: continue
                cnt=hi-lo
                colBase = (lo-d) if yForward else (lo+d)
                if diag=='U' and d==0:
                    for t in range(cnt): acc[lo-r0+t]+=x[colBase+t]
                    continue
                aColBase=colBase
                if isT: aColBase = (colBase+d) if uplo=='U' else (colBase-d)
                for t in range(cnt):
                    br = aRow-blockRowLo          # row within staged block
                    bc = aColBase-colLo+t         # col within staged block
                    if not (0<=br<sliceRows): return None,f"block row OOB br={br} slice={sliceRows}"
                    if not (0<=bc<colHi-colLo):  return None,f"block col OOB bc={bc} span={colHi-colLo}"
                    a=A[blockRowLo+br, colLo+bc]
                    if trans=='C': a=np.conj(a)
                    acc[lo-r0+t]+=a*x[colBase+t]
        y[r0:r0+rows]=acc
    return y,None

rng=np.random.default_rng(5); bad=0; tot=0; firsts=[]
for n,k in [(8,2),(7,6),(16,3),(129,5),(9,8)]:
    for uplo,trans,diag in itertools.product('UL','NTC','NU'):
        for R,SL in ((4,32),(1000,32),(4,2),(1000,2)):
            A=rng.normal(size=(k+1,n))+1j*rng.normal(size=(k+1,n))
            x=rng.normal(size=n)+1j*rng.normal(size=n)
            r=ref(A,x,n,k,uplo,trans,diag); g,err=kernel(A,x,n,k,uplo,trans,diag,R,SL); tot+=1
            if err or not np.allclose(r,g,atol=1e-10):
                bad+=1
                if len(firsts)<8: firsts.append(f"n={n} k={k} {uplo}{trans}{diag} R={R} SL={SL}: {err or 'value mismatch'}")
print(f"{tot} configs, {bad} bad")
for f in firsts: print("  ",f)
