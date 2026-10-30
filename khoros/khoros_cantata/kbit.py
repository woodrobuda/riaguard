"""Bit-level helpers needed by historical Khoros data handling."""





from __future__ import annotations
import numpy as np


def pack_bits(bits, lsb=False):
    b=np.asarray(bits,dtype=np.uint8).ravel()
    out=np.zeros((b.size+7)//8,dtype=np.uint8)
    for i,v in enumerate(b):
        if v:
            out[i//8] |= (1 << (i%8)) if lsb else (0x80 >> (i%8))
    return out

def unpack_bits(packed, nbits=None, lsb=False):
    p=np.asarray(packed,dtype=np.uint8).ravel()
    if nbits is None: nbits=p.size*8
    out=np.zeros(nbits,dtype=np.uint8)
    for i in range(nbits):
        out[i]=((p[i//8] >> (i%8 if lsb else 7-(i%8))) & 1)
    return out

def get_bit(packed, index, lsb=False):
    p=np.asarray(packed,dtype=np.uint8).ravel()
    return int((p[index//8] >> (index%8 if lsb else 7-index%8)) & 1)

def set_bit(packed, index, value, lsb=False):
    p=np.asarray(packed,dtype=np.uint8)
    mask=1 << (index%8 if lsb else 7-index%8)
    if value: p[index//8] |= mask
    else: p[index//8] &= np.uint8(~mask & 0xff)
    return p
