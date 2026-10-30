"""Turn a parsed Khoros pane into a simple operator specification."""





from dataclasses import dataclass,field
from .kforms_compat import parse_file
@dataclass
class OperatorProperty:
 variable:str;kind:str;title:str|None;description:str|None;default:object=None;bounds:object=None;options:list|None=None;optional:bool=False;metadata:dict=field(default_factory=dict)
@dataclass
class OperatorSpec:
 toolbox:str;name:str;pane_path:str;program:str|None;properties:list[OperatorProperty]
 @classmethod
 def from_document(cls,tb,name,path,doc):
  props=[];prog=None
  for r in doc.selections():
   f=r.fields
   if r.kind=='routine':prog=f.get('routine')
   elif r.variable:props.append(OperatorProperty(r.variable,r.kind,f.get('title'),f.get('description'),f.get('default',f.get('value')),f.get('bounds'),f.get('options'),bool(f.get('optional',0)),dict(f)))
  return cls(tb,name,path,prog,props)
