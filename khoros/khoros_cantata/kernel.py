"""Core graph engine used by the Python-native Cantata runtime. A workspace is a flowchart: Node is a box, Port is a plug, Connection is a line, and DataObject is data moving along the line. This module does not start a historical Khoros program."""





from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import Any, Callable, Optional, Protocol
import operator
import re

class NodeState(Enum):
    STOPPED = auto(); SCHEDULED = auto(); RUNNING = auto(); ERROR = auto()

class WorkspaceState(Enum):
    IDLE = auto(); RUNNING = auto(); STOPPING = auto(); COMPLETE = auto(); ERROR = auto()

class EventType(Enum):
    DAV = auto(); START = auto(); COMPLETE = auto(); ERROR = auto(); STOP = auto(); CHILD_COMPLETE = auto()

@dataclass
class Event:
    type: EventType
    node_id: str | None = None
    port_id: str | None = None
    payload: Any = None

@dataclass
class DataObject:
    value: Any = None
    temporary: bool = True
    protected: bool = False
    modified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

class Transport(Protocol):
    permanent: bool
    def put(self, data: DataObject) -> str: ...
    def get(self, token: str) -> DataObject: ...
    def delete(self, token: str) -> None: ...

class MemoryTransport:
    permanent = False
    def __init__(self): self._data: dict[str, DataObject] = {}; self._n = 0
    def put(self, data: DataObject) -> str:
        self._n += 1; token = f"mem:{self._n}"; self._data[token] = data; return token
    def get(self, token: str) -> DataObject: return self._data[token]
    def delete(self, token: str) -> None: self._data.pop(token, None)

@dataclass
class Connection:
    output: 'Port'; input: 'Port'; token: str | None = None

@dataclass
class Port:
    id: str
    node: 'Node'
    direction: str
    dav: bool = False
    data: DataObject | None = None
    connection: Connection | None = None

    def publish(self, data: DataObject):
        self.data = data; self.dav = True
        if self.connection:
            self.connection.input.data = data
            self.connection.input.dav = True
            ws = self.node.workspace
            if ws: ws.events.append(Event(EventType.DAV, self.node.id, self.id, self.connection.input.id))

    def consume(self) -> Any:
        if not self.dav or self.data is None: raise RuntimeError(f"data unavailable on {self.id}")
        return self.data.value

@dataclass
class Node:
    id: str
    inputs: dict[str, Port] = field(default_factory=dict)
    outputs: dict[str, Port] = field(default_factory=dict)
    state: NodeState = NodeState.STOPPED
    modified: bool = True
    error: Exception | None = None
    executor: Optional[Callable[['Node'], None]] = None
    workspace: 'Workspace | None' = None
    context: dict[str, Any] = field(default_factory=dict)

    def ready(self) -> bool:
        if self.state is not NodeState.STOPPED or not self.modified: return False
        if any(not p.dav for p in self.inputs.values()): return False
        for p in self.inputs.values():
            if p.connection:
                src = p.connection.output.node
                if src.state is NodeState.RUNNING or src.modified: return False
        return True

    def execute(self):
        self.state = NodeState.RUNNING
        if self.workspace: self.workspace.events.append(Event(EventType.START, self.id))
        try:
            if self.executor: self.executor(self)
            self.modified = False
            self.state = NodeState.STOPPED
            if self.workspace: self.workspace.events.append(Event(EventType.COMPLETE, self.id))
        except Exception as exc:
            self.error = exc; self.modified = True; self.state = NodeState.ERROR
            if self.workspace: self.workspace.events.append(Event(EventType.ERROR, self.id, payload=exc))
            raise

    def reset(self):
        self.state = NodeState.STOPPED; self.error = None; self.modified = True
        for p in [*self.inputs.values(), *self.outputs.values()]: p.dav = False; p.data = None

    def stop(self):
        self.state = NodeState.STOPPED

class ProcedureNode(Node):
    def __init__(self, id: str, child: 'Workspace', **kwargs):
        super().__init__(id, **kwargs); self.child = child
        child.parent_node = self

    def execute(self):
        self.state = NodeState.RUNNING
        if self.workspace: self.workspace.events.append(Event(EventType.START, self.id))
        try:
            ok = self.child.run()
            if not ok: raise self.child.error or RuntimeError(f"child workspace failed: {self.id}")
            self.modified = False; self.state = NodeState.STOPPED
            if self.workspace: self.workspace.events.append(Event(EventType.CHILD_COMPLETE, self.id))
        except Exception as exc:
            self.error = exc; self.modified = True; self.state = NodeState.ERROR
            if self.workspace: self.workspace.events.append(Event(EventType.ERROR, self.id, payload=exc))
            raise

    def reset(self): super().reset(); self.child.reset()
    def stop(self): super().stop(); self.child.stop()

class ConditionalNode(Node):
    """Graph-control node: choose one branch and publish control/data availability."""
    def __init__(self, id: str, predicate: Callable[['Node'], bool], **kwargs):
        super().__init__(id, **kwargs); self.predicate = predicate
    def execute(self):
        self.state = NodeState.RUNNING
        try:
            chosen = bool(self.predicate(self))
            self.context['branch'] = chosen
            self.modified = False; self.state = NodeState.STOPPED
            for name, p in self.outputs.items():
                p.dav = chosen if name in ('true','yes','then') else (not chosen if name in ('false','no','else') else True)
            if self.workspace: self.workspace.events.append(Event(EventType.COMPLETE, self.id, payload=chosen))
        except Exception as exc:
            self.error = exc; self.state = NodeState.ERROR
            if self.workspace: self.workspace.events.append(Event(EventType.ERROR, self.id, payload=exc))
            raise

class LoopNode(ProcedureNode):
    def __init__(self, id: str, child: 'Workspace', condition: Callable[[dict[str, Any]], bool], count: int | None = None, **kwargs):
        super().__init__(id, child, **kwargs); self.condition = condition; self.count = count
    def execute(self):
        self.state = NodeState.RUNNING
        try:
            i = 0
            while (self.count is None or i < self.count) and self.condition(self.context | {'iteration': i}):
                self.child.reset()
                if not self.child.run(): raise self.child.error or RuntimeError(f"loop child failed: {self.id}")
                i += 1
            self.context['iterations'] = i
            self.modified = False; self.state = NodeState.STOPPED
            if self.workspace: self.workspace.events.append(Event(EventType.CHILD_COMPLETE, self.id, payload=i))
        except Exception as exc:
            self.error = exc; self.modified = True; self.state = NodeState.ERROR
            if self.workspace: self.workspace.events.append(Event(EventType.ERROR, self.id, payload=exc))
            raise

class ExpressionContext:
    """Compatibility-oriented nested scope; intentionally not Python eval."""
    def __init__(self, parent: 'ExpressionContext | None' = None): self.values: dict[str, Any] = {}; self.parent = parent
    def get(self, name: str) -> Any:
        if name in self.values: return self.values[name]
        if self.parent: return self.parent.get(name)
        raise NameError(name)
    def set(self, name: str, value: Any): self.values[name] = value; return value

_TOKEN = re.compile(r'\s*(?:(\d+(?:\.\d+)?)|([A-Za-z_]\w*)|(==|!=|<=|>=|&&|\|\||[()+\-*/%<>!=]))')
class ExpressionEngine:
    """Small safe expression subset; later phases can extend toward full KExpr grammar."""
    def __init__(self, context: ExpressionContext): self.ctx = context
    def evaluate(self, text: str) -> Any:
        toks=[]; pos=0
        while pos < len(text):
            m=_TOKEN.match(text,pos)
            if not m: raise SyntaxError(f"unsupported expression near {text[pos:]!r}")
            toks.append(next(g for g in m.groups() if g is not None)); pos=m.end()
        if not toks: return None
        if len(toks)>=3 and toks[1]=='=' and re.fullmatch(r'[A-Za-z_]\w*', toks[0]):
            v=self._parse(toks[2:]); return self.ctx.set(toks[0],v)
        return self._parse(toks)
    def _parse(self,toks):
        # Pratt parser for arithmetic/comparison/logical operators.
        prec={'||':1,'&&':2,'==':3,'!=':3,'<':4,'>':4,'<=':4,'>=':4,'+':5,'-':5,'*':6,'/':6,'%':6}
        i=0
        def primary():
            nonlocal i
            t=toks[i]; i+=1
            if t=='(':
                v=expr(0)
                if toks[i]!=')': raise SyntaxError('missing )')
                i+=1; return v
            if re.fullmatch(r'\d+(?:\.\d+)?',t): return float(t) if '.' in t else int(t)
            if t in ('!','-'):
                v=primary(); return (not v) if t=='!' else -v
            return self.ctx.get(t)
        def expr(minp):
            nonlocal i
            left=primary()
            while i<len(toks) and toks[i] in prec and prec[toks[i]]>=minp:
                op=toks[i]; p=prec[op]; i+=1; right=expr(p+1)
                left={'||':lambda a,b:bool(a) or bool(b),'&&':lambda a,b:bool(a) and bool(b),'==':operator.eq,'!=':operator.ne,'<':operator.lt,'>':operator.gt,'<=':operator.le,'>=':operator.ge,'+':operator.add,'-':operator.sub,'*':operator.mul,'/':operator.truediv,'%':operator.mod}[op](left,right)
            return left
        v=expr(0)
        if i!=len(toks): raise SyntaxError('trailing tokens')
        return v

class Workspace:
    def __init__(self, transport: Transport | None = None, parent_node: Node | None = None, parent_context: ExpressionContext | None = None):
        self.nodes: dict[str, Node] = {}; self.frontier: deque[str] = deque(); self.events: deque[Event] = deque()
        self.transport = transport or MemoryTransport(); self.state = WorkspaceState.IDLE; self.error: Exception | None = None
        self.parent_node = parent_node; self.expression_context = ExpressionContext(parent_context); self._stopped = False

    @property
    def running(self): return self.state is WorkspaceState.RUNNING

    def add(self, node: Node): node.workspace = self; self.nodes[node.id] = node; return node

    def connect(self, out: Port, inp: Port):
        if inp.connection: raise ValueError(f"input already connected: {inp.id}")
        c = Connection(out, inp); out.connection = c; inp.connection = c
        c.token = self.transport.put(DataObject())
        return c

    def schedule(self, node: Node):
        if node.state is NodeState.STOPPED and node.modified and node.id not in self.frontier:
            node.state = NodeState.SCHEDULED; self.frontier.append(node.id)

    def request_dispatch(self):
        for n in self.nodes.values():
            if n.ready(): self.schedule(n)

    def dispatch(self):
        self.request_dispatch()
        while self.frontier and self.state is WorkspaceState.RUNNING and not self._stopped:
            node = self.nodes[self.frontier.popleft()]
            if node.state is not NodeState.SCHEDULED: continue
            try: node.execute()
            except Exception as exc:
                self.error = exc; self.state = WorkspaceState.ERROR; return
            self.request_dispatch()

    def run(self):
        if self.state is WorkspaceState.RUNNING: return False
        self.error = None; self._stopped = False; self.state = WorkspaceState.RUNNING
        self.dispatch()
        if self.state is WorkspaceState.RUNNING:
            unresolved = [n.id for n in self.nodes.values() if n.modified and n.state is not NodeState.ERROR]
            if unresolved: self.error = RuntimeError(f"workspace stalled; unresolved nodes: {unresolved}"); self.state = WorkspaceState.ERROR
            else: self.state = WorkspaceState.COMPLETE
        return self.state is WorkspaceState.COMPLETE

    def check(self):
        return {n.id: n.ready() for n in self.nodes.values()}

    def reset(self):
        self.frontier.clear(); self.events.clear(); self.error = None; self.state = WorkspaceState.IDLE; self._stopped = False
        for n in self.nodes.values(): n.reset()
        self.expression_context.values.clear()

    def stop(self):
        self._stopped = True; self.state = WorkspaceState.STOPPING; self.frontier.clear()
        for n in self.nodes.values(): n.stop()
        self.state = WorkspaceState.IDLE

    def cleanup(self):
        self.frontier.clear(); self.events.clear(); self.expression_context.values.clear()
        for c in self.connections():
            if c.token and not self.transport.permanent: self.transport.delete(c.token)
            c.token = None

    def connections(self):
        seen=set()
        for n in self.nodes.values():
            for p in n.outputs.values():
                if p.connection and id(p.connection) not in seen:
                    seen.add(id(p.connection)); yield p.connection
