"""Read and write the historical VIFF data format used by Khoros. VIFF can hold several logical data segments and metadata. read_viff() is the main entry point."""





from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .kdbm_compat import *
from .kdbm_compat import _machine_endian

VERSION = '2.0b'
COOKIE = 'ThisIsAViff'
VERSION_KEY='Version'; COOKIE_KEY='ViffMagicCookie'

@dataclass
class ViffSegment:
    name: str
    attributes: dict = field(default_factory=dict)
    data: object = None
    datatype: int | None = None
    size: tuple = ()
    dimension: int = 0
    index_order: tuple = ()
    record_meta: dict = field(default_factory=dict)

@dataclass
class ViffObject:
    attributes: dict = field(default_factory=dict)
    segments: dict[str,ViffSegment] = field(default_factory=dict)
    machine_code: int = 8
    source_path: str | None = None
    record_meta: dict = field(default_factory=dict)

    @property
    def version(self): return self.attributes.get('Version')
    @property
    def cookie(self): return self.attributes.get('ViffMagicCookie')

    def segment(self,name):
        if name not in self.segments: self.segments[name]=ViffSegment(name)
        return self.segments[name]

def _typed_record(num,size,dtype,payload,endian):
    import struct
    return struct.pack(endian+'3i',int(num),int(size),int(dtype))+payload

def _parse_record(raw,endian):
    if len(raw)<12: raise KDBMFormatError('VIFF record shorter than typed header')
    num,size,dtype=__import__('struct').unpack_from(endian+'3i',raw,0)
    return num,size,dtype,decode_typed(raw[12:],num,size,dtype,endian)

def read_viff(path) -> ViffObject:
    db=KDBM.open(path)
    if db.get(VERSION_KEY) != VERSION.encode()+b'\0':
        raise KDBMFormatError('not a Khoros 2.0b VIFF version record')
    cookie=db.get(COOKIE_KEY)
    if cookie != COOKIE.encode()+b'\0': raise KDBMFormatError('bad VIFF magic cookie')
    obj=ViffObject(machine_code=db.header.machine_code, source_path=str(path))
    obj.attributes.update({VERSION_KEY:VERSION, COOKIE_KEY:COOKIE})
    pending_data=[]
    for ent in db.entries():
        key=ent.key.rstrip(b'\0').decode('utf-8','replace')
        if key in (VERSION_KEY,COOKIE_KEY): continue
        raw=ent.data
        if ':' in key:
            segname,attr=key.split(':',1)
            if segname == '':
                if len(raw)<12: raise KDBMFormatError(f'bad typed VIFF object attribute {key}')
                num,size,dtype,val=_parse_record(raw,db.header.endian)
                obj.attributes[attr]=val
                obj.record_meta[attr]=(num,size,dtype)
                continue
            seg=obj.segment(segname)
            if attr:
                if len(raw)<12: raise KDBMFormatError(f'bad typed VIFF attribute {key}')
                num,size,dtype,val=_parse_record(raw,db.header.endian)
                seg.attributes[attr]=val
                seg.record_meta[attr]=(num,size,dtype)
                if attr=='dimension': seg.dimension=int(np.asarray(val).ravel()[0])
                elif attr=='size': seg.size=tuple(int(x) for x in np.asarray(val).ravel())
                elif attr=='index_order': seg.index_order=tuple(int(x) for x in np.asarray(val).ravel())
                elif attr=='data_type': seg.datatype=int(np.asarray(val).ravel()[0])
            else:
                pending_data.append((seg,raw))
        else:
            if len(raw)>=12:
                num,size,dtype,val=_parse_record(raw,db.header.endian)
                obj.attributes[key]=val
                obj.record_meta[key]=(num,size,dtype)
            else:
                obj.attributes[key]=raw
    for seg,raw in pending_data:
        if seg.datatype is None: raise KDBMFormatError(f'no data_type for VIFF segment {seg.name}')
        seg.data=decode_typed(raw,1,int(np.prod(seg.size)) if seg.size else len(raw),seg.datatype,db.header.endian)
        if seg.size and seg.datatype != KBIT:
            arr=np.asarray(seg.data)
            if arr.size==int(np.prod(seg.size)): seg.data=arr.reshape(seg.size)
    return obj

def _infer_khoros_type(a):
    a=np.asarray(a)
    if a.dtype==np.uint8: return KUBYTE
    if a.dtype==np.int8: return KBYTE
    if a.dtype==np.int16: return KSHORT
    if a.dtype==np.uint16: return KUSHORT
    if a.dtype==np.int32: return KINT
    if a.dtype==np.uint32: return KUINT
    if a.dtype==np.float32: return KFLOAT
    if a.dtype==np.float64: return KDOUBLE
    if a.dtype==np.complex64: return KCOMPLEX
    if a.dtype==np.complex128: return KDCOMPLEX
    raise TypeError(f'unsupported VIFF dtype {a.dtype}')

def _attr_record(value, dtype, endian):
    a=np.asarray(value) if dtype!=KSTRING else value
    if dtype==KSTRING:
        vals=[value] if isinstance(value,str) else list(value)
        payload=encode_typed(vals,KSTRING,endian); return _typed_record(len(vals),1,dtype,payload,endian)
    if dtype==KBIT:
        arr=np.asarray(value,dtype=np.uint8).ravel(); bits=arr.size
        payload=encode_typed(arr,KBIT,endian); return _typed_record(1,bits,dtype,payload,endian)
    arr=np.asarray(value); payload=encode_typed(arr.ravel(),dtype,endian)
    return _typed_record(1,arr.size,dtype,payload,endian)

def write_viff(path, obj: ViffObject, machine_code=None):
    if machine_code is None: machine_code=obj.machine_code
    endian=_machine_endian(machine_code)
    records=[]
    records.append((VERSION_KEY,VERSION.encode()+b'\0'))
    records.append((COOKIE_KEY,COOKIE.encode()+b'\0'))
    for key,val in obj.attributes.items():
        if key in (VERSION_KEY,COOKIE_KEY): continue
        meta=obj.record_meta.get(key)
        if meta is not None:
            num,size,dtype=meta
            if dtype==KSTRING:
                payload=encode_typed([val] if isinstance(val,str) else val,dtype,endian)
            else:
                payload=encode_typed(np.asarray(val).ravel(),dtype,endian)
            records.append(((':'+key) if not key.startswith(':') else key,__import__('struct').pack(endian+'3i',num,size,dtype)+payload))
        else:
            dtype=KSTRING if isinstance(val,str) or (isinstance(val,(list,tuple)) and val and isinstance(val[0],str)) else _infer_khoros_type(val)
            records.append(((':'+key) if not key.startswith(':') else key,_attr_record(val,dtype,endian)))
    for name,s in obj.segments.items():
        # Special physical attributes are always persisted by Viff.c.
        if s.data is not None:
            arr=np.asarray(s.data)
            dtype=s.datatype or _infer_khoros_type(arr)
            records.append((f'{name}:',encode_typed(arr.ravel(),dtype,endian)))
        dim=s.dimension or (len(s.size) if s.size else (np.asarray(s.data).ndim if s.data is not None else 0))
        size=s.size or (tuple(np.asarray(s.data).shape) if s.data is not None else ())
        order=s.index_order or tuple(range(dim))
        if dim:
            records.append((f'{name}:dimension',_attr_record([dim],KINT,endian)))
            records.append((f'{name}:data_type',_attr_record([s.datatype or _infer_khoros_type(np.asarray(s.data))],KINT,endian)))
            records.append((f'{name}:size',_attr_record(size,KINT,endian)))
            records.append((f'{name}:index_order',_attr_record(order,KINT,endian)))
        for attr,val in s.attributes.items():
            if attr in ('dimension','data_type','size','index_order'): continue
            meta=s.record_meta.get(attr)
            if meta is not None:
                num,size,dtype=meta
                payload=encode_typed([val] if isinstance(val,str) else val,dtype,endian) if dtype==KSTRING else encode_typed(np.asarray(val).ravel(),dtype,endian)
                records.append((f'{name}:{attr}',__import__('struct').pack(endian+'3i',num,size,dtype)+payload))
            else:
                dtype=KSTRING if isinstance(val,str) or (isinstance(val,(list,tuple)) and all(isinstance(x,str) for x in val)) else _infer_khoros_type(val)
                records.append((f'{name}:{attr}',_attr_record(val,dtype,endian)))
    db=KDBM.write_records(path,records,machine_code=machine_code)
    return db

def viff_to_kdms(v: ViffObject):
    """Materialize a recovered VIFF into the project's KDMSObject model."""
    from .kdms_object import KDMSObject
    dtype_map={KBYTE:np.int8,KUBYTE:np.uint8,KSHORT:np.int16,KUSHORT:np.uint16,KINT:np.int32,KUINT:np.uint32,
               KLONG:np.int32,KULONG:np.uint32,KFLOAT:np.float32,KDOUBLE:np.float64,KCOMPLEX:np.complex64,KDCOMPLEX:np.complex128}
    o=KDMSObject(fmt='viff',architecture=v.machine_code)
    for k,val in v.attributes.items():
        if k not in (VERSION_KEY,COOKIE_KEY): o.set_attribute(None,k,val)
    for name,s in v.segments.items():
        o.create_segment(name)
        dt=dtype_map.get(s.datatype,np.uint8)
        o.set_attribute(name,'data_type',dt)
        if s.size: o.set_attribute(name,'size',s.size)
        if s.index_order: o.set_attribute(name,'index_order',s.index_order)
        for k,val in s.attributes.items():
            if k not in ('dimension','data_type','size','index_order'): o.set_attribute(name,k,val)
        if s.data is not None:
            o.put_data(name,np.asarray(s.data,dtype=dt))
    return o
