"""Read the historical Khoros KDBM database records needed by this project. This is a focused compatibility reader, not a complete GDBM replacement."""





from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math, struct
import numpy as np

MAGIC = 0x13579ACF
OLD_MAGIC = 0x13579ACE
BUCKET_AVAIL = 6
SMALL = 4
BUCKET_ELEMS = 48
BLOCK_SIZE = 1024

# Khoros datatype flags.
KBIT=1; KBYTE=2; KUBYTE=4; KSHORT=8; KUSHORT=16; KINT=32; KUINT=64
KLONG=128; KULONG=256; KFLOAT=512; KDOUBLE=1024; KCOMPLEX=2048
KDCOMPLEX=4096; KSTRING=8192; KSTRUCT=16384; KLOGICAL=65536

@dataclass
class KDBMHeader:
    machine_code: int
    endian: str
    long_size: int
    header_magic: int
    block_size: int
    dir_size: int
    dir_bits: int
    bucket_size: int
    bucket_elems: int
    next_block: int
    dir: int

@dataclass
class KDBMEntry:
    key: bytes
    data: bytes
    hash_value: int
    bucket_address: int
    slot: int

class KDBMFormatError(ValueError):
    pass

def _machine_endian(code: int) -> str:
    # Khoros kdbm.c maps these GDBM machine identifiers.
    return '<' if code in (1,2,20,23) else '>'

def _machine_long_size(code: int) -> int:
    # The Alpha historical database uses 64-bit longs; the other recovered
    # Khoros KDBM variants use 32-bit longs in their on-disk representation.
    return 8 if code == 23 else 4

def khash(key: bytes) -> int:
    # Historical hash_dbm(). key.dsize includes the terminating NUL for Khoros strings.
    value = (0x238F13AF * len(key)) & 0x7FFFFFFF
    for i, ch in enumerate(key):
        value = (value + (ch << ((i*5) % 24))) & 0x7FFFFFFF
    return (1103515243 * value + 12345) & 0x7FFFFFFF

def _u32(b, off, endian): return struct.unpack_from(endian+'I', b, off)[0]
def _i32(b, off, endian): return struct.unpack_from(endian+'i', b, off)[0]

def _pack(fmt, *vals): return struct.pack(fmt, *vals)

def _dtype_info(t):
    return {
        KBYTE: ('b', np.int8), KUBYTE: ('B', np.uint8), KSHORT: ('h', np.int16),
        KUSHORT: ('H', np.uint16), KINT: ('i', np.int32), KUINT: ('I', np.uint32),
        KLONG: ('i', np.int32), KULONG: ('I', np.uint32), KFLOAT: ('f', np.float32),
        KDOUBLE: ('d', np.float64), KCOMPLEX: ('ff', np.complex64),
        KDCOMPLEX: ('dd', np.complex128), KLOGICAL: ('i', np.int32),
    }.get(t)

def decode_typed(payload: bytes, num: int, size: int, dtype: int, endian: str):
    if dtype == KSTRING:
        out=[]; p=0
        for _ in range(num*size):
            q=payload.find(b'\0',p)
            if q < 0: q=len(payload)
            out.append(payload[p:q].decode('utf-8','replace')); p=q+1
        return out
    if dtype == KBIT:
        return np.frombuffer(payload, dtype=np.uint8).copy()
    info=_dtype_info(dtype)
    if info is None:
        return payload
    fmt, ndt=info
    count=num*size
    if dtype in (KLONG,KULONG) and len(payload) >= count*8 and len(payload) == count*8:
        fmt='q' if dtype==KLONG else 'Q'; ndt=np.int64 if dtype==KLONG else np.uint64
    if dtype in (KCOMPLEX,KDCOMPLEX):
        vals=np.frombuffer(payload, dtype=np.dtype(endian+fmt[0]), count=count*2).copy()
        return (vals[0::2] + 1j*vals[1::2]).astype(ndt)
    arr=np.frombuffer(payload, dtype=np.dtype(endian+fmt[0]), count=count).copy()
    return arr.astype(ndt, copy=False)

def encode_typed(values, dtype: int, endian: str):
    if dtype == KSTRING:
        if isinstance(values, str): values=[values]
        return b''.join(str(x).encode('utf-8')+b'\0' for x in values)
    if dtype == KBIT:
        return np.asarray(values, dtype=np.uint8).tobytes()
    info=_dtype_info(dtype)
    if info is None: return bytes(values)
    fmt, ndt=info
    if dtype in (KLONG,KULONG) and np.asarray(values).dtype.itemsize >= 8:
        fmt='q' if dtype==KLONG else 'Q'; ndt=np.int64 if dtype==KLONG else np.uint64
    arr=np.asarray(values)
    if dtype in (KCOMPLEX,KDCOMPLEX):
        base=np.float32 if dtype==KCOMPLEX else np.float64
        x=np.asarray(arr, dtype=np.complex64 if dtype==KCOMPLEX else np.complex128)
        inter=np.empty(x.size*2,dtype=base); inter[0::2]=x.real; inter[1::2]=x.imag
        return inter.astype(np.dtype(endian+('f' if base==np.float32 else 'd')),copy=False).tobytes()
    return np.asarray(arr,dtype=ndt).astype(np.dtype(endian+fmt[0]),copy=False).tobytes()

class KDBM:
    def __init__(self, path, data: bytes):
        self.path=Path(path) if path is not None else None
        self.data=data
        if len(data)<64: raise KDBMFormatError('file too small for KDBM')
        self.header=self._read_header()
        self.directory=self._read_directory()
        self._entries=None

    def _read_header(self):
        code=self.data[0]
        endian=_machine_endian(code)
        magic=struct.unpack_from(endian+'i',self.data,1)[0]
        if magic==OLD_MAGIC: raise KDBMFormatError('pre-Khoros-2.0 KDBM')
        if magic!=MAGIC: raise KDBMFormatError(f'bad KDBM magic 0x{magic:08x}')
        magic,block,ds,db,bs,be=struct.unpack_from(endian+'6i',self.data,1)
        ls=_machine_long_size(code); off=1+6*4
        next_block=struct.unpack_from(endian+('q' if ls==8 else 'i'),self.data,off)[0]; off+=ls
        dir_=struct.unpack_from(endian+('q' if ls==8 else 'i'),self.data,off)[0]
        return KDBMHeader(code,endian,ls,magic,block,ds,db,bs,be,next_block,dir_)

    def _read_directory(self):
        h=self.header; n=h.dir_size//h.long_size
        if h.dir<0 or h.dir+n*h.long_size>len(self.data): raise KDBMFormatError('directory outside file')
        fmt='q' if h.long_size==8 else 'i'
        return list(struct.unpack_from(h.endian+f'{n}{fmt}',self.data,h.dir))

    def _bucket(self, addr):
        h=self.header; p=addr; e=h.endian
        if p<0 or p+h.bucket_size>len(self.data): raise KDBMFormatError('bucket outside file')
        av_count,bits,count=struct.unpack_from(e+'3i',self.data,p); p+=12
        p+=BUCKET_AVAIL*2*h.long_size
        rows=[]
        for i in range(h.bucket_elems):
            hv,ks,ds=struct.unpack_from(e+'3i',self.data,p); p+=12
            ptr=struct.unpack_from(e+('q' if h.long_size==8 else 'i'),self.data,p)[0]; p+=h.long_size
            key_start=self.data[p:p+SMALL]; p+=SMALL
            rows.append((hv,ks,ds,ptr,key_start))
        return bits,count,rows

    def entries(self):
        if self._entries is not None: return list(self._entries)
        out=[]; seen=set()
        for addr in self.directory:
            if addr in seen: continue
            seen.add(addr)
            bits,count,rows=self._bucket(addr)
            for slot,(hv,ks,ds,ptr,ks4) in enumerate(rows):
                if hv==-1: continue
                if ptr<0 or ptr+ks+ds>len(self.data): raise KDBMFormatError('entry outside file')
                key=self.data[ptr:ptr+ks]; val=self.data[ptr+ks:ptr+ks+ds]
                out.append(KDBMEntry(key,val,hv,addr,slot))
        self._entries=out
        return list(out)

    def get(self,key):
        if isinstance(key,str): key=key.encode()+b'\0'
        hv=khash(key); h=self.header
        di=hv >> (31-h.dir_bits)
        addr=self.directory[di]
        _,_,rows=self._bucket(addr)
        loc=hv%h.bucket_elems; start=loc
        while rows[loc][0]!=-1:
            rh,ks,ds,ptr,_=rows[loc]
            if rh==hv and ks==len(key) and self.data[ptr:ptr+ks]==key:
                return self.data[ptr+ks:ptr+ks+ds]
            loc=(loc+1)%h.bucket_elems
            if loc==start: break
        return None

    def typed(self,key):
        raw=self.get(key)
        if raw is None: return None
        if len(raw)<12: raise KDBMFormatError('typed record too short')
        e=self.header.endian
        num,size,dtype=struct.unpack_from(e+'3i',raw,0)
        return num,size,dtype,decode_typed(raw[12:],num,size,dtype,e)

    def as_dict(self):
        return {x.key.rstrip(b'\0').decode('utf-8','replace'): x.data for x in self.entries()}

    @classmethod
    def open(cls,path): return cls(path,Path(path).read_bytes())

    @classmethod
    def write_records(cls,path, records, block_size=None, machine_code=8):
        """Create a readable Khoros-2.0 KDBM database in the historical layout."""
        records=list(records); e=_machine_endian(machine_code); ls=_machine_long_size(machine_code)
        if block_size is None: block_size=2048 if ls==8 else 1024
        if block_size % ls: raise KDBMFormatError('block size must be long-aligned')
        norm=[]; seen=set()
        for key,val in records:
            key=(key.encode()+b'\0') if isinstance(key,str) else bytes(key); val=bytes(val)
            if key in seen: raise KDBMFormatError(f'duplicate KDBM key {key!r}')
            seen.add(key); norm.append((key,val,khash(key)))
        # Directory size is grown from 8 longs until it reaches the I/O block.
        dir_size=8*ls; dir_bits=3
        while dir_size<block_size: dir_size*=2; dir_bits+=1
        bucket_elems=48; bucket_size=block_size
        buckets={0:norm}; depths={0:0}; next_bid=1
        while True:
            target=next((bid for bid,items in buckets.items() if len(items)>bucket_elems),None)
            if target is None: break
            depth=depths[target]
            if depth>=dir_bits: raise KDBMFormatError('unable to split KDBM bucket at directory depth')
            items=buckets.pop(target); depths.pop(target); nd=depth+1; bitpos=30-depth
            b0=[]; b1=[]
            for rec in items: (b1 if ((rec[2]>>bitpos)&1) else b0).append(rec)
            b0id=target; b1id=next_bid; next_bid+=1
            buckets[b0id]=b0; buckets[b1id]=b1; depths[b0id]=nd; depths[b1id]=nd
        bucket_items=list(buckets.items()); bucket_addr={bid:(2+i)*block_size for i,(bid,_) in enumerate(bucket_items)}
        directory=[]
        for di in range(dir_size//ls):
            chosen=None
            for bid,items in bucket_items:
                d=depths[bid]
                if d==0: chosen=bid; break
                prefix=di >> (dir_bits-d); hp=items[0][2] >> (31-d) if items else None
                if hp==prefix: chosen=bid; break
            if chosen is None: raise KDBMFormatError('internal directory construction error')
            directory.append(bucket_addr[chosen])
        bucket_count=len(bucket_items); data_start=(2+bucket_count)*block_size; blob=bytearray(data_start)
        fmt_long='q' if ls==8 else 'i'
        header=bytearray(block_size); header[0]=machine_code
        struct.pack_into(e+'6i',header,1,MAGIC,block_size,dir_size,dir_bits,bucket_size,bucket_elems)
        struct.pack_into(e+fmt_long,header,1+6*4,0)
        struct.pack_into(e+fmt_long,header,1+6*4+ls,block_size)
        # avail.size/count/next and first pair. The remainder is unused in a fresh closed file.
        struct.pack_into(e+'2i',header,1+6*4+2*ls,122,0)
        struct.pack_into(e+fmt_long,header,1+6*4+2*ls+8,0)
        struct.pack_into(e+fmt_long,header,1+6*4+3*ls+8,0)
        blob[:block_size]=header
        # Directory block.
        off=block_size
        blob[off:off+dir_size]=struct.pack(e+f'{len(directory)}{fmt_long}',*directory)
        next_ptr=data_start
        row_size=12+ls+SMALL; bucket_prefix=12+BUCKET_AVAIL*2*ls
        for bid,items in bucket_items:
            bucket=bytearray(block_size); struct.pack_into(e+'3i',bucket,0,0,depths[bid],len(items)); used=set()
            for i in range(bucket_elems):
                q=bucket_prefix+i*row_size; struct.pack_into(e+'3i',bucket,q,-1,0,0); struct.pack_into(e+fmt_long,bucket,q+12,0)
            for key,val,hv in items:
                slot=hv%bucket_elems; start=slot
                while slot in used:
                    slot=(slot+1)%bucket_elems
                    if slot==start: raise KDBMFormatError('KDBM bucket overflow')
                used.add(slot); ptr=next_ptr; blob.extend(key+val); next_ptr+=len(key)+len(val)
                q=bucket_prefix+slot*row_size; struct.pack_into(e+'3i',bucket,q,hv,len(key),len(val)); struct.pack_into(e+fmt_long,bucket,q+12,ptr); bucket[q+12+ls:q+12+ls+SMALL]=key[:4].ljust(4,b'\0')
            addr=bucket_addr[bid]; blob[addr:addr+block_size]=bucket
        struct.pack_into(e+fmt_long,blob,1+6*4,next_ptr)
        Path(path).write_bytes(blob); return cls.open(path)

