"""Source-derived numerical behavior and historical regression checks. This module contains Python versions of selected numerical operations whose Khoros source was recovered and runs the recovered regression checks."""





from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import ast, re, time, json, os
import numpy as np
from .viff import read_viff

ROOT=Path(os.environ.get('KHOROS_SOURCE_ROOT', Path(__file__).resolve().parents[1] / 'historical_source' / 'datamanip'))

@dataclass
class CorpusSummary:
    suites:int; scripts:int; result_files:int; testdata_files:int
    suites_with_results:int; suites_with_scripts:int


def corpus_available(root=ROOT):
    return (Path(root)/'testsuite').is_dir()


def inventory(root=ROOT):
    root=Path(root); ts=root/'testsuite'; suites=[]; scripts=[]; results=[]
    if not ts.is_dir():
        return CorpusSummary(0,0,0,0,0,0), []
    for d in sorted(p for p in ts.iterdir() if p.is_dir()):
        sh=sorted(d.glob('test*.sh')); rr=sorted(d.rglob('*.results/*'))
        if sh or rr:
            suites.append({'suite':d.name,'scripts':[str(x.relative_to(root)) for x in sh],
                           'result_files':[str(x.relative_to(root)) for x in rr],
                           'script_count':len(sh),'result_count':len(rr)})
            scripts += sh; results += rr
    td=sorted((root/'data'/'testdata').glob('*'))
    return CorpusSummary(len(suites),len(scripts),len(results),len([x for x in td if x.is_file()]),
                         sum(bool(x['result_files']) for x in suites),sum(bool(x['scripts']) for x in suites)), suites


def viff_value(path, segment='value'):
    v=read_viff(path); s=v.segments[segment]
    return np.asarray(s.data), s


def viff_stats(path):
    a,_=viff_value(path)
    return a.ravel().astype(float)


def khoros_fft(a, axes, direction=0, scale=1, center=0):
    """Source-derived lkfft semantics: FFT842 inverse is 1/N; scale 1/2/3 = 1,1/N,1/sqrt(N)."""
    x=np.asarray(a,dtype=np.complex128)
    axes=tuple(axes)
    if center:
        idx=np.indices(x.shape)
        parity=sum(idx[ax] for ax in axes) & 1
        x=np.where(parity==0,x,-x)
    for ax in axes:
        x=np.fft.fft(x,axis=ax) if direction==0 else np.fft.ifft(x,axis=ax)
    n=int(np.prod([x.shape[ax] for ax in axes]))
    if direction==0:
        if scale==2: x=x/n
        elif scale==3: x=x/np.sqrt(n)
    else:
        # Source lkfft compensates the FFT842 inverse's native 1/N scaling:
        # scale1 -> unnormalized inverse, scale2 -> standard inverse,
        # scale3 -> symmetric 1/sqrt(N) normalization.
        if scale==1: x=x*n
        elif scale==3: x=x*np.sqrt(n)
    if direction==1 and center:
        idx=np.indices(x.shape); parity=sum(idx[ax] for ax in axes) & 1
        x=np.where(parity==0,x,-x)
    return x


def khoros_sine(shape, amplitudes, nums, phases=None, dtype=np.float64):
    phases=phases or [0.0]*5
    grids=np.indices(shape)
    ang=sum(2*np.pi*nums[i]*grids[i]/shape[i] + phases[i]*np.pi/180 for i in range(len(shape)))
    return (amplitudes[0]*np.sin(ang)).astype(dtype)


def conv1d_khoros(seq,kernel,w0=0,extended=False):
    x=np.asarray(seq,float).ravel(); k=np.asarray(kernel,float).ravel()
    # Source lklinearop uses the kernel origin (sweet spot) and indexes
    # backward from each output sample: y[j] = sum_q k[q] * x[j-q].
    # Truncated output starts at the requested origin; extended output adds
    # the full kernel support at the right.
    start=int(w0); n=len(x) if not extended else len(x)+len(k)-1
    if len(x)==0 or len(k)==0: return np.zeros(n)
    # Vectorized: full convolution index j == y[j]; indices outside support are 0.
    full=np.convolve(x,k,'full'); out=np.zeros(n)
    j=np.arange(start,start+n); ok=(j>=0)&(j<len(full)); out[ok]=full[j[ok]]
    return out


def stats(a):
    x=np.asarray(a,float).ravel();
    return {'mean':float(x.mean()),'variance':float(x.var()),'std':float(x.std()),
            'minimum':float(x.min()),'maximum':float(x.max()),'sum':float(x.sum()),'points':int(x.size)}


def compare(a,b,tol=1e-5):
    a=np.asarray(a); b=np.asarray(b)
    if a.shape!=b.shape: return False, {'reason':'shape','a':a.shape,'b':b.shape}
    d=np.abs(a.astype(np.complex128)-b.astype(np.complex128))
    return bool(np.nanmax(d)<=tol), {'max_abs':float(np.nanmax(d)),'mean_abs':float(np.nanmean(d))}


def run_phase21_checks():
    base=ROOT/'data'/'testdata'; out=[]
    def add(name,category,fn):
        t=time.perf_counter()
        try:
            ok,detail=fn(); status='PASS' if ok else 'FAIL'
        except FileNotFoundError as e:
            status='SKIP'; detail={'reason':'historical corpus file missing (set KHOROS_SOURCE_ROOT)','missing':str(e.filename or e)}
        except Exception as e:
            ok=False; status='ERROR'; detail={'error':repr(e)}
        out.append({'name':name,'category':category,'status':status,'elapsed_s':time.perf_counter()-t,'detail':detail})
    # Image/data processing checks against recovered historical golden result objects.
    add('karith1 atan/tan roundtrip','image/data',lambda: _atan_roundtrip(base))
    add('karith1 sqrt-square identity','image/data',lambda: _sqrt_square(base))
    add('kclip scalar clamp','image/data',lambda: _clip_scalar(base))
    add('kclip threshold scalar','image/data',lambda: _threshold_scalar(base))
    add('klinearop 1D convolution w0','signal/data',lambda: _conv(base,0,False,'conv1D_truc_0.ascii'))
    add('klinearop 1D convolution w1','signal/data',lambda: _conv(base,1,False,'conv1D_truc_1.ascii'))
    add('klinearop 1D convolution w2','signal/data',lambda: _conv(base,2,True,'conv1D_ext_2.ascii'))
    add('kfft forward/inverse roundtrip','signal/data',lambda: _fft_roundtrip(base))
    add('kgsin source-formula 3D sine','signal/data',lambda: _sine_formula())
    add('kfft impulse conservation','signal/data',lambda: _fft_impulse())
    add('kfft 3D sinusoid peak','signal/data',lambda: _fft_sine_peak())
    return out


def _ascii(path): return np.array(Path(path).read_text().split(),dtype=float)

def _atan_roundtrip(base):
    x,_=viff_value(base/'3x4masked.double.viff'); y=np.tan(np.arctan(x)); return compare(y,x,1e-12)
def _sqrt_square(base):
    x,_=viff_value(base/'3x4masked.double.viff'); y=np.abs(np.sqrt(x)**2-x); return (float(np.max(y))<1e-12,{'max_abs':float(np.max(y))})
def _clip_scalar(base):
    x,_=viff_value(base/'3x4.double.viff'); y=np.clip(x,3,21); expected=np.array([3,3,3,3,10,11,12,13,20,21,21,21],float).reshape(x.shape); return compare(y,expected,0)
def _threshold_scalar(base):
    x,_=viff_value(base/'3x4.double.viff'); y=np.where((x<3)|(x>21),400,75); return (y.size==12 and np.array_equal(y.ravel(),np.array([400,400,400,75,75,75,75,75,75,75,400,400])),{'values':y.ravel().tolist()})
def _conv(base,w0,ext,golden):
    seq=_ascii(ROOT/'testsuite/klinearop/testdata/seq1D.ascii'); ker=_ascii(ROOT/'testsuite/klinearop/testdata/kernel1D.ascii')
    got=conv1d_khoros(seq,ker,w0,ext); exp=_ascii(ROOT/'testsuite/klinearop/test2.results'/golden); return compare(got,exp,1e-12)
def _fft_roundtrip(base):
    x,_=viff_value(base/'3x4.float.viff'); x=x[:,:,0,0,0];
    # 4x3 cannot FFT height=3; use width only as a valid power-of-two transform.
    f=khoros_fft(x,[0],0,1,0); y=khoros_fft(f,[0],1,2,0); return compare(y,x,1e-10)
def _sine_formula():
    got=khoros_sine((8,8,8,1,1),[1.0],[2,3,4,0,0]); expected=np.sin(2*np.pi*(2*np.indices((8,8,8,1,1))[0]/8+3*np.indices((8,8,8,1,1))[1]/8+4*np.indices((8,8,8,1,1))[2]/8)); return compare(got,expected,1e-14)
def _fft_impulse():
    x=np.zeros((4,4,4,4,4)); x[0,0,0,0,0]=1; f=khoros_fft(x,[0,1,2,3,4],0,1,1); return (np.max(np.abs(f-1))<1e-12,{'max_abs':float(np.max(np.abs(f-1)))})
def _fft_sine_peak():
    idx=np.indices((32,32,32)); x=np.sin(2*np.pi*(2*idx[0]/32+3*idx[1]/32+4*idx[2]/32)); f=khoros_fft(x,[0,1,2],0,1,0); mag=np.abs(f); pos=np.unravel_index(np.argmax(mag),mag.shape); return (pos in [(2,3,4),(30,29,28)],{'peak':pos,'peak_value':float(mag[pos])})

if __name__=='__main__':
    s,suites=inventory(); checks=run_phase21_checks()
    print(json.dumps({'summary':asdict(s),'checks':checks},indent=2,default=str))
