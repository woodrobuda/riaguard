"""Model the fields shown by a Khoros operator property panel."""





from dataclasses import dataclass
@dataclass
class PropertyField:
 variable:str;kind:str;title:str|None;description:str|None;value:object;default:object;bounds:object=None;options:list|None=None;optional:bool=False;metadata:dict|None=None
class PropertyPanelModel:
 def __init__(self,spec):self.spec=spec;self.fields=[PropertyField(p.variable,p.kind,p.title,p.description,p.default,p.default,p.bounds,p.options,p.optional,p.metadata) for p in spec.properties]
 def set(self,v,x):
  for f in self.fields:
   if f.variable==v:f.value=x;return
  raise KeyError(v)
 def get(self,v):
  for f in self.fields:
   if f.variable==v:return f.value
  raise KeyError(v)
