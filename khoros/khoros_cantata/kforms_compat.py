"""Parse historical Khoros UIS and pane files. A pane describes operator controls. This parser turns those records into a semantic Python tree."""





from dataclasses import dataclass,field
from pathlib import Path
import re,shlex
T={'F':'startform','S':'startmaster','D':'startsubmenu','M':'startsubform','d':'subformbutton','u':'pseudo_subform','n':'masteraction','G':'startguide','P':'startpane','g':'guidebutton','Q':'quit','m':'subformaction','H':'help','I':'inputfile','O':'outputfile','e':'stdin','o':'stdout','i':'integer','f':'float','h':'double','l':'logical','t':'flag','s':'string','y':'stringlist','R':'routine','T':'toggle','b':'blank','a':'paneaction','c':'cycle','x':'list','z':'displaylist','p':'includepane','k':'includesubform','C':'mutex_group','B':'inclusive_group','K':'group','w':'workspace','E':'end'}
SEL={'inputfile','outputfile','stdin','stdout','integer','float','double','logical','flag','string','stringlist','routine','toggle','cycle','list','displaylist','workspace'}
CONT={'startform','startmaster','startsubmenu','startsubform','startguide','startpane','mutex_group','inclusive_group','group','toggle'}
GEOM=re.compile(r'^[-+]?\d+(?:\.\d+)?x[-+]?\d+(?:\.\d+)?[+-][-+]?\d+(?:\.\d+)?[+-][-+]?\d+(?:\.\d+)?$')
def atom(x):
 if x in ('true','TRUE','True'): return True
 if x in ('false','FALSE','False'): return False
 try:return int(x) if re.fullmatch(r'[+-]?\d+',x) else float(x) if re.fullmatch(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?',x) else x
 except:return x
@dataclass
class UISRecord:
 code:str; kind:str; raw:str; tokens:list[str]; fields:dict=field(default_factory=dict); children:list=field(default_factory=list); parent:object=None; line_no:int=0
 @property
 def variable(self):return self.fields.get('variable')
@dataclass
class KFormsDocument:
 path:str|None; records:list; roots:list; comments:list=field(default_factory=list); errors:list=field(default_factory=list)
 def selections(self):
  out=[]
  def walk(rs):
   for r in rs:
    if r.kind in SEL:out.append(r)
    walk(r.children)
  walk(self.roots);return out
 def property_schema(self):
  return [{'variable':r.variable,'kind':r.kind,'title':r.fields.get('title'),'description':r.fields.get('description'),'default':r.fields.get('default',r.fields.get('value')),'value':r.fields.get('value'),'bounds':r.fields.get('bounds'),'options':r.fields.get('options'),'optional':bool(r.fields.get('optional',0)),'optional_selection':r.fields.get('opt_sel'),'live':r.fields.get('live'),'raw':r.raw} for r in self.selections()]
 def consolidate_args(self,argv):
  vars={r.variable:r for r in self.selections() if r.variable};out=[];i=0
  while i<len(argv):
   a=argv[i];r=vars.get(a[1:]) if a.startswith('-') else None
   if r:
    i+=1
    if r.kind not in {'flag','toggle'} and i<len(argv) and not argv[i].startswith('-'):i+=1
   else:out.append(a);i+=1
  return out
 def selected_command(self,include_program=True):
  args=[];program=None;stdin=stdout=None
  for r in self.selections():
   f=r.fields;v=f.get('value');var=r.variable
   if r.kind=='routine': program=f.get('routine');continue
   if not f.get('selected',False):continue
   if r.kind=='stdin':stdin=f.get('filename')
   elif r.kind=='stdout':stdout=f.get('filename')
   elif var:
    args.append('-'+var)
    if r.kind=='flag':pass
    elif r.kind=='logical':args.append('1' if bool(v) else '0')
    elif r.kind in {'cycle','list','displaylist'}:args.append(str(f.get('selection_index',v if v is not None else 0)))
    elif r.kind=='toggle':
     if f.get('toggle_value') is not None:args.append(str(f['toggle_value']))
    elif v is not None:args.append(str(v))
  out=[]
  if stdin:out+=['kcp','-i',stdin,'-o','-','|']
  if include_program and program:out.append(program)
  out+=args
  if stdout:out+=['|','kcp','-i','-','-o',stdout,'-force']
  return out

def parse_record(line,no):
 raw=line; s=line.strip()
 if not s:return UISRecord('','blankline',raw,[],line_no=no)
 if s.startswith('#'):return UISRecord('#','comment',raw,[s],line_no=no)
 tok=shlex.split(s,posix=True)
 if not tok or not tok[0].startswith('-'):return UISRecord('','metadata',raw,tok,line_no=no)
 c=tok[0][1:];r=UISRecord(c,T.get(c,'unknown'),raw,tok,line_no=no);f=r.fields
 if c=='E':return r
 if c in 'BC':f['group_value']=atom(tok[1]) if len(tok)>1 else None;return r
 if c=='K':f['logical_val']=atom(tok[1]) if len(tok)>1 else None;return r
 if c=='u':f['obsolete']=True;f['unsupported_reason']='Khoros 2.0 createtree.c rejects pseudo-subform buttons';return r
 if c=='S':f.update(activate=atom(tok[1]),selected=atom(tok[2]));return r
 if c=='F':f.update(version=tok[1],activate=atom(tok[2]),selected=atom(tok[3]),geometry=tok[4],secondary_geometry=tok[5],title=tok[6],variable=tok[7]);return r
 if c in 'MP':f.update(activate=atom(tok[1]),selected=atom(tok[2]),geometry=tok[3],secondary_geometry=tok[4],title=tok[5],variable=tok[6]);return r
 if c in 'Dg':f.update(activate=atom(tok[1]),selected=atom(tok[2]),geometry=tok[3],title=tok[4],variable=tok[5]);return r
 if c=='G':f.update(activate=atom(tok[1]),geometry=tok[2],secondary_geometry=tok[3],title=tok[4]);return r
 if c=='Q':f.update(activate=atom(tok[1]),selected=atom(tok[2]),geometry=tok[3],title=tok[4]);return r
 if c in 'dnma':f.update(activate=atom(tok[1]),selected=atom(tok[2]),geometry=tok[3],title=tok[4],description=tok[5],variable=tok[6]);return r
 if c=='H':f.update(activate=atom(tok[1]),geometry=tok[2],title=tok[3],description=tok[4],help_file=tok[5],variable=tok[6]);return r
 if c in 'IO':
  f.update(activate=atom(tok[1]),selected=atom(tok[2]),optional=atom(tok[3]),opt_sel=atom(tok[4]),live=atom(tok[5]),file_check=atom(tok[6]),geometry=tok[7],default=tok[8],title=tok[9],description=tok[10],variable=tok[11]);
  if len(tok)>12:f['filename']=tok[12]
  if len(tok)>13:f['literal']=tok[13]
  return r
 if c in 'ifh':
  f.update(activate=atom(tok[1]),selected=atom(tok[2]),optional=atom(tok[3]),opt_sel=atom(tok[4]),live=atom(tok[5]),geometry=tok[6]); nums=[atom(x) for x in tok[7:12 if c=='i' else 13]];f['bounds']=nums[:2];f['default']=nums[2] if len(nums)>2 else None;f['value']=f['default'];f['numeric_tail']=nums[3:];q=12 if c=='i' else 13;f['title']=tok[q];f['description']=tok[q+1];f['variable']=tok[q+2];f['literal']=tok[q+3] if len(tok)>q+3 else None;return r
 if c=='l':f.update(activate=atom(tok[1]),selected=atom(tok[2]),optional=atom(tok[3]),opt_sel=atom(tok[4]),live=atom(tok[5]),geometry=tok[6],default=atom(tok[7]),label0=tok[8],label1=tok[9],description=tok[10],variable=tok[11],value=atom(tok[12]));return r
 if c=='t':f.update(activate=atom(tok[1]),selected=atom(tok[2]),optional=atom(tok[3]),opt_sel=atom(tok[4]),live=atom(tok[5]),geometry=tok[6],title=tok[7],description=tok[8],variable=tok[9],value=bool(atom(tok[2])));return r
 if c=='s':f.update(activate=atom(tok[1]),selected=atom(tok[2]),optional=atom(tok[3]),opt_sel=atom(tok[4]),live=atom(tok[5]),geometry=tok[6],default=tok[7],title=tok[8],description=tok[9],variable=tok[10],value=tok[7]);f['literal']=tok[11] if len(tok)>11 else None;return r
 if c=='y':f.update(activate=atom(tok[1]),selected=atom(tok[2]),optional=atom(tok[3]),opt_sel=atom(tok[4]),live=atom(tok[5]),geometry=tok[6],num_options=atom(tok[7]),default_index=atom(tok[8]),title=tok[9],description=tok[10],variable=tok[11],value=atom(tok[8]));return r
 if c=='R':f.update(activate=atom(tok[1]),selected=atom(tok[2]),exec_type=atom(tok[3]),geometry=tok[4],title=tok[5],description=tok[6],routine=' '.join(tok[7:]));return r
 if c=='T':f.update(activate=atom(tok[1]),selected=atom(tok[2]),optional=atom(tok[3]),opt_sel=atom(tok[4]),live=atom(tok[5]),geometry=tok[6],secondary_geometry=tok[7],default=atom(tok[8]),title=tok[9],description=tok[10],variable=tok[11],toggle_value=atom(tok[8]));return r
 if c in 'eo':f.update(activate=atom(tok[1]),optional=atom(tok[2]),opt_sel=atom(tok[3]),geometry=tok[4],title=tok[5],variable=tok[6],filename=tok[7] if len(tok)>7 else None);return r
 if c in 'cxz':
  f.update(activate=atom(tok[1]),selected=atom(tok[2]),optional=atom(tok[3]),opt_sel=atom(tok[4]),live=atom(tok[5]),geometry=tok[6]); n=4 if c=='z' else 3; nums=[atom(x) for x in tok[7:7+n]];f.update(num_options=nums[0],start=nums[1],default_index=nums[2],selection_index=nums[2]);idx=7+n
  if c=='z':f['double_click']=nums[3]
  f.update(title=tok[idx],description=tok[idx+1],variable=tok[idx+2],options=tok[idx+3:]);f['value']=f['default_index'];return r
 if c=='b':f.update(position=tok[1],description=tok[2],variable=tok[3]);return r
 if c in 'pk':f['filename']=tok[1];return r
 if c=='w':f.update(geometry=f'{tok[1]}x{tok[2]}+{tok[3]}+{tok[4]}',description=tok[5],variable=tok[6],workspace_address=atom(tok[7]) if len(tok)>7 else None);return r
 return r

def parse_uis(text,path=None,strict=True):
 rec=[];roots=[];stack=[];comments=[];errors=[]
 for no,line in enumerate(text.splitlines(),1):
  if not line.strip():continue
  r=parse_record(line,no);rec.append(r)
  if r.kind in {'comment','metadata'}:comments.append((no,line));continue
  if r.code=='E':
   if stack:stack.pop()
   elif strict:raise ValueError(f'line {no}: unmatched -E')
   continue
  (stack[-1].children if stack else roots).append(r)
  if stack:r.parent=stack[-1]
  if r.kind in CONT:stack.append(r)
 if stack and strict:raise ValueError('unterminated containers')
 return KFormsDocument(path,rec,roots,comments,errors)
def parse_file(path,strict=True):
 p=Path(path);return parse_uis(p.read_text(errors='replace'),str(p),strict)
