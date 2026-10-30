"""Read Khoros Configuration Management System records. This module reads toolbox and software-object records but does not run the executable named by a record."""





from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from .kdbm_compat import KDBM

PROG_TYPES={
 'kroutine':1, 'xvroutine':2, 'example':4, 'xexample':8, 'testsuite':16,
 'library':32, 'script':64, 'pane':128, 'manual':256, 'mindex':512,
 'mglossary':1024, 'dir':2048, 'mdir':4096,
}

@dataclass
class SoftwareObject:
    name:str
    object_type:str=''
    attributes:dict[str,object]=field(default_factory=dict)
    @property
    def prog_type(self): return self.attributes.get('PROG_TYPE',self.object_type)
    @property
    def uis_pane(self): return self.attributes.get('UIS_PANE')
    @property
    def uis_form(self): return self.attributes.get('UIS_FORM')
    @property
    def workspace(self): return self.attributes.get('WORKSPACE')

@dataclass
class Toolbox:
    name:str
    attributes:dict[str,object]=field(default_factory=dict)
    software_objects:list[SoftwareObject]=field(default_factory=list)
    def object(self,name):
        for o in self.software_objects:
            if o.name==name: return o
        return None

def _decode(raw):
    if raw is None: return None
    return raw.rstrip(b'\0').decode('utf-8','replace')

def read_cms(path, name=None):
    db=KDBM.open(path)
    attrs={}
    for e in db.entries(): attrs[e.key.rstrip(b'\0').decode('utf-8','replace')]=_decode(e.data)
    return attrs

def read_toolbox_db(path, name=None):
    attrs=read_cms(path,name)
    tb=Toolbox(name or attrs.get('TB_NAME',''))
    tb.attributes=attrs
    raw=attrs.get('SOFTWARE_OBJECTS','')
    for line in raw.splitlines():
        line=line.strip()
        if not line: continue
        if ':' in line: oname,otype=line.split(':',1)
        else: oname,otype=line,''
        tb.software_objects.append(SoftwareObject(oname,otype))
    return tb

def read_software_object(path,name):
    attrs=read_cms(path,name)
    return SoftwareObject(name,attrs.get('PROG_TYPE',''),attrs)

def load_toolbox(root):
    root=Path(root)
    tb=read_toolbox_db(root/'repos/db/cms',root.name)
    for o in tb.software_objects:
        p=root/'objects'
        candidates=list(p.rglob(o.name))
        cms=None
        for c in candidates:
            q=c/'db/cms'
            if q.is_file(): cms=q; break
        if cms: o2=read_software_object(cms,o.name); o.attributes.setdefault('OBJECT_PATH',str(c)); o.__dict__.update(o2.__dict__)
    return tb

def write_cms(path, attributes: dict, machine_code=8, block_size=None):
    """Write a KCMS-style raw-string KDBM database."""
    from .kdbm_compat import KDBM
    records=[]
    for key,val in attributes.items():
        raw=val if isinstance(val,(bytes,bytearray)) else str(val).encode('utf-8')+b'\0'
        records.append((key,raw))
    return KDBM.write_records(path,records,machine_code=machine_code,block_size=block_size)
