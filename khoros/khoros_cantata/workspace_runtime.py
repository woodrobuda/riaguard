"""Read and run historical Cantata workspace files. A workspace describes boxes, connections, and variables. The runner builds a Python graph and uses a Python-native operator registry. It never passes old command text to a shell."""





from __future__ import annotations
import ast, copy, shlex, re, types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from .kernel import Workspace, Node, Port, ProcedureNode, LoopNode, ConditionalNode, DataObject, NodeState

@dataclass
class HistoricalGlyph:
    toolbox: str
    oname: str
    name: str
    ident: int
    x: int
    y: int
    variables: str = ''
    command: str = ''

@dataclass
class HistoricalConnection:
    out_id: int
    out_port: str
    in_id: int
    in_port: str

@dataclass
class HistoricalWorkspace:
    glyphs: list[HistoricalGlyph] = field(default_factory=list)
    connections: list[HistoricalConnection] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    annotations: list[str] = field(default_factory=list)

class HistoricalWorkspaceParser:
    """Parser for the two workspace record dialects observed in supplied Khoros sources.

    Release-era examples use both the explicit `Glyph:` form and the older compact
    `Connection:` form.  The parser deliberately preserves raw command text because
    commands are an observable part of WorkSave/WorkRestore semantics.
    """
    def parse(self, source: str | Path) -> HistoricalWorkspace:
        text = Path(source).read_text(encoding='utf-8') if isinstance(source,(str,Path)) else source
        lines=text.splitlines(); out=HistoricalWorkspace(); i=0; mode=None
        while i < len(lines):
            line=lines[i].strip(); i+=1
            if not line: continue
            if line == '#GlyphBegin' or line == '#ConnectionBegin':
                mode='glyph' if 'Glyph' in line else 'connection'; continue
            # End markers must be handled before the generic comment skip, otherwise the
            # glyph mode leaks past #GlyphEnd and NodeConnection lines are swallowed.
            if line in ('#ConnectionEnd','#GlyphEnd'): mode=None; continue
            if line.startswith('#') and not line.startswith('#Annotations') and not line.startswith('#Variables'):
                continue
            if line == '#AnnotationsBegin':
                while i<len(lines) and lines[i].strip()!='#AnnotationsEnd':
                    if lines[i].strip(): out.annotations.append(lines[i].rstrip())
                    i+=1
                if i<len(lines): i+=1
                continue
            if line == 'VariablesBegin:':
                while i<len(lines) and lines[i].strip()!='VariablesEnd:':
                    s=lines[i].strip(); i+=1
                    if not s or s.startswith('#') or '=' not in s: continue
                    k,v=s.split('=',1)
                    try: out.variables[k]=ast.literal_eval(v)
                    except Exception:
                        try: out.variables[k]=float(v) if '.' in v else int(v)
                        except Exception: out.variables[k]=v
                if i<len(lines): i+=1
                continue
            if line.startswith('Glyph:'):
                # Khoros 2.x cantata form: Glyph:TB:ONAME:LABEL:ID:X:Y:VARS:COMMAND
                # (LABEL usually empty, e.g. "Glyph:DATAMANIP:kfft::7:185:225:: -dir 0").
                q=line.split(':',8)
                if len(q)>=8 and not q[3].lstrip('-').isdigit() and q[4].lstrip('-').isdigit():
                    try:
                        out.glyphs.append(HistoricalGlyph(q[1],q[2],q[3] or q[2],int(q[4]),int(q[5]),int(q[6]),q[7],q[8] if len(q)>8 else ''))
                        continue
                    except ValueError: pass
                p=line.split(':',8)
                if len(p)>=8:
                    _,tb,oname,ident,x,y,variables,rest = p[:8] if len(p)==8 else (p[0],p[1],p[2],p[3],p[4],p[5],p[6],':'.join(p[7:]))
                    # Explicit historical form: Glyph:TB:ONAME:ID:X:Y:VARS:COMMAND
                    if len(p)>=8:
                        try:
                            out.glyphs.append(HistoricalGlyph(tb,oname,oname,int(ident),int(x),int(y),variables,rest))
                        except ValueError: pass
                continue
            if line.startswith('Conditional:'):
                p=line.split(':',8)
                if len(p)>=8:
                    try: out.glyphs.append(HistoricalGlyph(p[1],p[2],p[3],int(p[4]),int(p[5]),int(p[6]),p[7],p[8] if len(p)>8 else ''))
                    except ValueError: pass
                continue
            if mode=='glyph' and not line.startswith(('Glyph:','Conditional:')) and ':' in line:
                p=line.split(':',6)
                if len(p)>=6:
                    try:
                        tb,oname,ident,x,y,cmd=p[0],p[1],p[2],p[3],p[4],p[5]
                        out.glyphs.append(HistoricalGlyph(tb,oname,oname,int(ident),int(x),int(y),'',cmd))
                    except ValueError:
                        pass
                continue
            if line.startswith('Connection:') or (mode=='connection' and re.match(r'^\d+:',line)):
                s=line[len('Connection:'):] if line.startswith('Connection:') else line
                p=s.split(':')
                if len(p)>=4:
                    try: out.connections.append(HistoricalConnection(int(p[0]),p[1],int(p[2]),p[3]))
                    except ValueError: pass
                continue
            if line.startswith('NodeConnection:'):
                p=line.split(':')
                if len(p)>=5:
                    try: out.connections.append(HistoricalConnection(int(p[1]),p[2],int(p[3]),p[4]))
                    except ValueError: pass
        return out

class NativeOperator:
    def __init__(self, fn: Callable[[dict[str,Any],dict[str,Any]],dict[str,Any]], inputs=None, outputs=None):
        self.fn=fn; self.inputs=inputs or []; self.outputs=outputs or ['o']
    def __call__(self, params, inputs): return self.fn(params,inputs)

class NativeOperatorRegistry:
    def __init__(self): self._ops: dict[tuple[str,str],NativeOperator]={}; self._register_defaults()
    def register(self,tb,oname,fn,inputs=None,outputs=None): self._ops[(tb.upper(),oname.lower())]=NativeOperator(fn,inputs,outputs)
    def resolve(self,tb,oname): return self._ops.get((tb.upper(),oname.lower())) or self._ops.get(('*',oname.lower()))
    def _register_defaults(self):
        import numpy as _np
        def _arr(x):
            # Scalars stay Python scalars (historical behavior); sequences become ndarrays.
            return _np.asarray(x) if isinstance(x,(list,tuple,_np.ndarray)) else x
        unary={
            'kneg':_np.negative,'kabs':_np.abs,'ksqr':_np.square,'ksqrt':_np.sqrt,
            'krecip':lambda x:_np.reciprocal(_np.asarray(x,dtype=float)),'kcbrt':_np.cbrt,'kcube':lambda x:_np.power(x,3),
            'kfloor':lambda x:_np.floor(x).astype(int),'ktruncate':lambda x:_np.trunc(x).astype(int),
            'kbitnot':lambda x:_np.invert(_np.asarray(x).astype(_np.int64))}
        def _wrap(f):
            def op(p,i):
                x=_arr(i['i']); r=f(x)
                if not isinstance(x,_np.ndarray) and _np.ndim(r)==0: r=r.item() if hasattr(r,'item') else r
                return {'o':r}
            return op
        for name,fn in unary.items(): self.register('DATAMANIP',name,_wrap(fn))
        for name,f in {'ksub':_np.subtract,'kadd':_np.add,'kmul':_np.multiply,'kdiv':_np.true_divide}.items():
            def op(p,i,f=f):
                a,b=_arr(i['i1']),_arr(i['i2']); r=f(a,b)
                if _np.ndim(r)==0 and not isinstance(a,_np.ndarray) and not isinstance(b,_np.ndarray): r=r.item()
                return {'o':r}
            self.register('DATAMANIP',name,op)
        self.register('DATAMANIP','kclip',self._clip)
        self.register('DATAMANIP','kclipout',self._clip)
        self.register('DATAMANIP','kmegaclip',lambda p,i:{'o':i.get('i')})
        def _c2r(p,i):
            x=_arr(i['i'])
            return {'om':_np.abs(x) if p.get('logmag1') else _np.real(x) if _np.iscomplexobj(x) else x}
        self.register('DATAMANIP','kcmplx2real',_c2r)
        self.register('*','identity',lambda p,i:{'o':i.get('i')})
        try:
            from .operators import register_all
            register_all(self)
        except ImportError:
            pass
    @staticmethod
    def _clip(p,i):
        """Clamp to [lc, uc]; either bound may be absent (None). A bound of 0 is a real bound."""
        import numpy as _np
        x=i['i']; lc=p.get('lc'); uc=p.get('uc')
        if lc is not None and uc is not None and lc>uc: raise ValueError('kclip: lower bound exceeds upper bound')
        def one(v):
            if lc is not None: v=max(lc,v)
            if uc is not None: v=min(uc,v)
            return v
        if isinstance(x,_np.ndarray):
            return {'o':_np.clip(x,lc,uc) if (lc is not None or uc is not None) else x.copy()}
        try: return {'o':[one(v) for v in x]}
        except TypeError: return {'o':one(x)}

def parse_command(command: str) -> tuple[list[str],dict[str,Any]]:
    """Extract executable basename and Khoros-style options without executing a shell."""
    if not command.strip(): return [],{}
    try: toks=shlex.split(command, posix=True)
    except ValueError: toks=command.split()
    # Locate program after an optional historical absolute path.
    program=''; args=[]; started=False
    for t in toks:
        if not started and not t.startswith('-'):
            program=Path(t).name; started=True; continue
        # Khoros 2.x workspaces store only the options (no program name).
        started=True; args.append(t)
    params={}; i=0
    while i<len(args):
        t=args[i]
        if t.startswith('-'):
            k=t.lstrip('-')
            nxt=args[i+1] if i+1<len(args) else None
            if nxt is not None and (not nxt.startswith('-') or re.fullmatch(r'-\d*\.?\d+([eE][-+]?\d+)?',nxt)): params[k]=nxt; i+=2
            else: params[k]=True; i+=1
        else: i+=1
    for k,v in list(params.items()):
        try: params[k]=ast.literal_eval(v) if isinstance(v,str) else v
        except Exception: pass
    return [program],params

class WorkspaceExecutor:
    """Restore and execute a historical graph using Python-native operators only."""
    def __init__(self, registry: NativeOperatorRegistry|None=None): self.registry=registry or NativeOperatorRegistry()
    def restore(self, hws: HistoricalWorkspace) -> Workspace:
        ws=Workspace(); nodes={}
        for g in hws.glyphs:
            op=self.registry.resolve(g.toolbox,g.oname)
            def make_exec(g=g,op=op,node_holder=None):
                def execute(n):
                    if op is None: raise RuntimeError(f'no native operator for {g.toolbox}:{g.oname}')
                    params=parse_command(g.command)[1]
                    vals={k:p.data.value for k,p in n.inputs.items() if p.dav and p.data is not None}
                    result=op(params,vals)
                    for name,val in result.items():
                        p=n.outputs.get(name) or Port(name,n,'output'); n.outputs[name]=p
                        p.publish(DataObject(val,temporary=True,modified=True))
                return execute
            n=Node(str(g.ident),executor=make_exec())
            n.context.update(toolbox=g.toolbox,oname=g.oname,name=g.name,x=g.x,y=g.y,variables=g.variables,command=g.command)
            ws.add(n); nodes[g.ident]=n
        # Historical Cantata permits one output port to fan out to many input ports.
        # The early Phase-3 kernel modeled only one connection per output, so Phase 20
        # installs a small compatibility fan-out at the workspace boundary.
        fanouts={}
        for c in hws.connections:
            if c.out_id not in nodes or c.in_id not in nodes: continue
            out=nodes[c.out_id].outputs.get(c.out_port) or Port(c.out_port,nodes[c.out_id],'output'); nodes[c.out_id].outputs[c.out_port]=out
            inp=nodes[c.in_id].inputs.get(c.in_port) or Port(c.in_port,nodes[c.in_id],'input'); nodes[c.in_id].inputs[c.in_port]=inp
            if out.connection is None:
                ws.connect(out,inp)
            else:
                from .kernel import Connection
                cc=Connection(out,inp); cc.token=ws.transport.put(DataObject()); inp.connection=cc
            fanouts.setdefault(id(out),[]).append(inp)
        for n in nodes.values():
            for out in n.outputs.values():
                targets=fanouts.get(id(out),[])
                if len(targets)>1:
                    original=out.publish
                    def publish(data, out=out, targets=targets, original=original):
                        original(data)
                        for inp in targets[1:]:
                            inp.data=data; inp.dav=True
                            ws.events.append(__import__(__package__ + '.kernel', fromlist=['Event']).Event(__import__(__package__ + '.kernel', fromlist=['EventType']).EventType.DAV,out.node.id,out.id,inp.id))
                    out.publish=publish
        for k,v in hws.variables.items(): ws.expression_context.set(k,v)
        return ws

    def execute(self,hws: HistoricalWorkspace) -> Workspace:
        ws=self.restore(hws); ok=ws.run()
        if not ok: raise ws.error or RuntimeError('workspace execution failed')
        return ws

# Small adapters used for historical example execution where the original sample-data
# files are absent. They deliberately generate deterministic data rather than pretending
# to reproduce the missing external files.
def example_registry() -> NativeOperatorRegistry:
    r=NativeOperatorRegistry()
    r.register('SAMPLEDATA','images',lambda p,i:{'image':[0,32,64,128,200,255]})
    r.register('SAMPLEDATA','rgbimages',lambda p,i:{'rgbimage':[0,32,64,128,200,255]})
    r.register('SAMPLEDATA','signals',lambda p,i:{'signal':[0.0,1.0,0.0,-1.0]})
    r.register('ENVISION','putimage',lambda p,i:{})
    r.register('ENVISION','puticon',lambda p,i:{})
    r.register('ENVISION','putplot2',lambda p,i:{})
    return r
