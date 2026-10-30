"""Python model of the main Khoros Data Services ideas. Khoros data can include value, location, time, mask, and map information. Keep those ideas separate."""





from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import copy
import numpy as np

ALL_SEGMENTS = "__ALL_SEGMENTS__"

@dataclass
class AttributeDef:
    name: str
    dimension: int = 1
    size: Tuple[int, ...] = (1,)
    dtype: Any = np.int32
    deletable: bool = False
    generic: bool = False
    default: Any = None

@dataclass
class PhysicalSegment:
    """Shared physical segment state; references share this object."""
    data: Optional[np.ndarray] = None
    dtype: Any = np.uint8
    index_order: Tuple[int, ...] = ()
    locked: bool = False
    temporary: bool = False
    permanence: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    attribute_defs: Dict[str, AttributeDef] = field(default_factory=dict)

@dataclass
class Segment:
    """Presentation of a physical segment.

    Khoros keeps physical attributes shared across references while presentation
    attributes belong to each reference.  The backing PhysicalSegment therefore
    must never be copied when kdms_reference() is used.
    """
    name: str
    physical: PhysicalSegment = field(default_factory=PhysicalSegment)
    data_type: Any = None
    presentation_order: Tuple[int, ...] = ()
    attributes: Dict[str, Any] = field(default_factory=dict)
    attribute_defs: Dict[str, AttributeDef] = field(default_factory=dict)

    @property
    def data(self):
        return self.physical.data
    @data.setter
    def data(self, value):
        self.physical.data = value

    @property
    def dtype(self):
        return self.physical.dtype
    @dtype.setter
    def dtype(self, value):
        self.physical.dtype = value

    @property
    def index_order(self):
        return self.presentation_order or self.physical.index_order
    @index_order.setter
    def index_order(self, value):
        self.presentation_order = tuple(value)

    @property
    def locked(self):
        return self.physical.locked
    @locked.setter
    def locked(self, value):
        self.physical.locked = value

    @property
    def permanence(self):
        return self.physical.permanence
    @permanence.setter
    def permanence(self, value):
        self.physical.permanence = bool(value)

    @property
    def dimension(self):
        return 0 if self.physical.data is None else self.physical.data.ndim

    @property
    def size(self):
        return () if self.physical.data is None else tuple(self.physical.data.shape)

class KDMSRegistry:
    """Registry for dynamically defined segment types and generic attributes."""
    def __init__(self):
        self.segment_types: Dict[str, Dict[str, Any]] = {}
        self.attribute_defs: Dict[Tuple[str, str], AttributeDef] = {}

    def define_segment(self, name, create=None, destroy=None, query=None, **hooks):
        if name in self.segment_types:
            return False
        self.segment_types[name] = {"create": create, "destroy": destroy,
                                    "query": query, **hooks}
        return True

    def undefine_segment(self, name):
        return self.segment_types.pop(name, None) is not None

    def define_attribute(self, segment, name, dimension, size, dtype,
                         deletable=False, generic=False, default=None):
        key = (segment, name)
        if key in self.attribute_defs:
            return False
        self.attribute_defs[key] = AttributeDef(name, dimension, tuple(size), dtype,
                                                deletable, generic, default)
        return True

GLOBAL_REGISTRY = KDMSRegistry()

class KDMSObject:
    BUILTIN_OBJECT_ATTRS = {
        "name": AttributeDef("name", dtype=str, deletable=False),
        "format": AttributeDef("format", dtype=str, deletable=False),
        "architecture": AttributeDef("architecture", dtype=np.int32, deletable=False),
    }
    BUILTIN_SEGMENT_ATTRS = {
        "dimension": AttributeDef("dimension", dtype=np.int32, deletable=False),
        "size": AttributeDef("size", dtype=np.int32, deletable=False),
        "data_type": AttributeDef("data_type", dtype=str, deletable=False),
        "index_order": AttributeDef("index_order", dtype=np.int32, deletable=False),
    }

    def __init__(self, name="", fmt="", architecture=0, registry=None):
        self.name, self.format, self.architecture = name, fmt, architecture
        self.registry = registry or GLOBAL_REGISTRY
        self.attributes: Dict[str, Any] = {}
        self.segments: Dict[str, Segment] = {}
        self.closed = False
        self._protected = False
        self._dirty = False

    def reference(self):
        """Create an independent presentation sharing all physical resources."""
        r = copy.copy(self)
        r.attributes = copy.deepcopy(self.attributes)
        r.segments = {}
        for name, src in self.segments.items():
            dst = Segment(name=name, physical=src.physical,
                          data_type=src.data_type,
                          presentation_order=tuple(src.presentation_order),
                          attributes=copy.deepcopy(src.attributes),
                          attribute_defs=copy.deepcopy(src.attribute_defs))
            r.segments[name] = dst
        r.closed = False
        r._protected = self._protected
        r._dirty = False
        refs = getattr(self, "_references", None)
        if refs is None:
            refs = []
            self._references = refs
        if not any(x is self for x in refs):
            refs.append(self)
        refs.append(r)
        r._references = refs
        return r

    def create_segment(self, name):
        if name in self.segments:
            return False
        # Historical segment-definition callbacks are allowed to call
        # kdms_create_segment() themselves.  Install the object first so the
        # callback can populate its attributes without recursive creation.
        s = Segment(name)
        self.segments[name] = s
        d = self.registry.segment_types.get(name)
        if d and d.get("create"):
            if not d["create"](self, name):
                self.segments.pop(name, None)
                return False
        return True

    def query_segment(self, name):
        return name in self.segments

    def destroy_segment(self, name):
        if name not in self.segments:
            return False
        d = self.registry.segment_types.get(name)
        if d and d.get("destroy"):
            if not d["destroy"](self, name):
                return False
        del self.segments[name]
        return True

    def set_attributes(self, segment=None, **attrs):
        if segment is None:
            for k, v in attrs.items(): self.set_attribute(None, k, v)
            return True
        if segment not in self.segments: self.create_segment(segment)
        s = self.segments[segment]
        for k, v in attrs.items():
            if k == "dimension":
                continue
            if k == "size":
                shape = tuple(int(x) for x in np.asarray(v).ravel())
                if any(x < 0 for x in shape):
                    return False
                dtype = s.dtype
                old = s.data
                if old is None:
                    s.data = np.zeros(shape, dtype=dtype)
                elif tuple(old.shape) != shape:
                    # Khoros resize preserves the existing physical region and
                    # initializes newly exposed space rather than discarding it.
                    new = np.zeros(shape, dtype=dtype)
                    common = tuple(slice(0, min(old.shape[i], shape[i]))
                                   for i in range(min(old.ndim, len(shape))))
                    if old.ndim == len(shape):
                        new[common] = old[common]
                    else:
                        return False
                    s.data = new
            elif k == "data_type":
                s.dtype = v
                if s.data is not None: s.data = s.data.astype(v)
            elif k == "index_order": s.index_order = tuple(int(x) for x in np.asarray(v).ravel())
            else: s.attributes[k] = v
        return True

    def set_attribute(self, segment, name, value):
        if segment is None: self.attributes[name] = value
        else:
            if segment not in self.segments: self.create_segment(segment)
            if name == "size":
                s = self.segments[segment]
                shape = tuple(int(x) for x in np.asarray(value).ravel())
                if any(x < 0 for x in shape):
                    return False
                old = s.data
                if old is None:
                    s.data = np.zeros(shape, dtype=s.dtype)
                elif tuple(old.shape) != shape:
                    if old.ndim != len(shape):
                        return False
                    new = np.zeros(shape, dtype=s.dtype)
                    common = tuple(slice(0, min(old.shape[i], shape[i])) for i in range(len(shape)))
                    new[common] = old[common]
                    s.data = new
            elif name == "data_type":
                s = self.segments[segment]
                s.dtype = value
                if s.data is not None:
                    s.data = s.data.astype(value)
            elif name == "index_order":
                self.segments[segment].index_order = tuple(int(x) for x in np.asarray(value).ravel())
            elif name == "permanence":
                self.segments[segment].permanence = bool(value)
            elif name == "dimension":
                # Dimension is represented by the size/order vectors in KDMS.
                self.segments[segment].attributes[name] = value
            else:
                self.segments[segment].attributes[name] = value
        self._dirty = True
        return True

    def get_attribute(self, segment, name, default=None):
        if segment is None:
            if name == "name": return self.name
            if name == "format": return self.format
            if name == "architecture": return self.architecture
            return self.attributes.get(name, default)
        if segment not in self.segments: return default
        s = self.segments[segment]
        if name == "dimension": return s.dimension
        if name == "size": return np.asarray(s.size, dtype=np.int32)
        if name == "data_type": return s.data_type if s.data_type is not None else np.dtype(s.dtype).name
        if name == "index_order": return np.asarray(s.index_order, dtype=np.int32)
        return s.attributes.get(name, default)

    def create_attribute(self, segment, name, dimension, size, dtype,
                         deletable=True, generic=True, default=None):
        scope = segment or ALL_SEGMENTS
        d = AttributeDef(name, dimension, tuple(size), dtype, deletable, generic, default)
        if segment is None:
            self.attributes[name] = default
        else:
            if segment not in self.segments: self.create_segment(segment)
            self.segments[segment].attribute_defs[name] = d
            self.segments[segment].attributes.setdefault(name, default)
        return True

    def define_attribute(self, segment, name, dimension, size, dtype,
                         deletable=False, generic=False, default=None):
        return self.registry.define_attribute(segment or ALL_SEGMENTS, name,
                                              dimension, size, dtype,
                                              deletable, generic, default)

    def destroy_attribute(self, segment, name):
        defs = self.BUILTIN_OBJECT_ATTRS if segment is None else self.BUILTIN_SEGMENT_ATTRS
        if name in defs: return False
        target = self.attributes if segment is None else self.segments.get(segment, Segment(segment)).attributes
        return target.pop(name, None) is not None

    def update_references(self, segment):
        """Propagate a segment presentation template to other references.

        Source: _kdms_update_references(). Physical resources remain shared;
        presentation metadata is copied to each reference.
        """
        if segment not in self.segments:
            return False
        template = self.segments[segment]
        for ref in getattr(self, "_references", []):
            if ref is self:
                continue
            src = ref.segments.get(segment)
            if src is None:
                src = Segment(name=segment, physical=template.physical)
                ref.segments[segment] = src
            src.data_type = copy.deepcopy(template.data_type)
            src.presentation_order = tuple(template.presentation_order)
            src.attributes = copy.deepcopy(template.attributes)
            src.attribute_defs = copy.deepcopy(template.attribute_defs)
        return True

    def copy_segment_attributes(self, source, source_segment, dest_segment):
        if source_segment is None:
            for k,v in source.attributes.items(): self.attributes[k] = copy.deepcopy(v)
        else:
            if source_segment not in source.segments: return False
            if dest_segment not in self.segments: self.create_segment(dest_segment)
            s, d = source.segments[source_segment], self.segments[dest_segment]
            d.attributes.update(copy.deepcopy(s.attributes))
            d.attribute_defs.update(copy.deepcopy(s.attribute_defs))
        return True

    def copy_segment_data(self, source, source_segment, dest_segment):
        """Copy one segment's physical data and transferable state."""
        if source_segment not in source.segments:
            return False
        if dest_segment not in self.segments and not self.create_segment(dest_segment):
            return False
        src = source.segments[source_segment]
        dst = self.segments[dest_segment]
        dst.physical.data = None if src.data is None else np.array(src.data, copy=True)
        dst.physical.dtype = src.physical.dtype
        dst.physical.index_order = tuple(src.physical.index_order)
        dst.data_type = copy.deepcopy(src.data_type)
        dst.presentation_order = tuple(src.presentation_order)
        return True

    def copy_object(self, source, include_nontransferable=False):
        """Copy transferable segments from *source* into this object."""
        for name, src in source.segments.items():
            transferable = src.attributes.get("transferable", True)
            if not transferable and not include_nontransferable:
                continue
            if not self.copy_segment_data(source, name, name):
                return False
            self.copy_segment_attributes(source, name, name)
        self._dirty = True
        return True

    def rename_segment(self, old, new):
        if old not in self.segments or new in self.segments:
            return False
        s = self.segments.pop(old)
        s.name = new
        self.segments[new] = s
        return True

    def get_data(self, segment, begin=None, end=None, dtype=None):
        s = self.segments[segment]
        if s.data is None: return np.empty((0,), dtype=dtype or s.dtype)
        a = s.data
        if begin is not None:
            b = tuple(int(x) for x in np.asarray(begin).ravel())
            e = tuple(int(x) for x in np.asarray(end).ravel())
            sl = tuple(slice(max(0,x), min(a.shape[i], e[i]+1)) for i,x in enumerate(b))
            a = a[sl]
        if dtype is not None: a = a.astype(dtype)
        return np.array(a, copy=True)

    def put_data(self, segment, data, begin=None, end=None):
        s = self.segments[segment]
        incoming = np.asarray(data)
        if s.data is None:
            s.data = np.array(incoming, copy=True)
        else:
            if self._protected and not getattr(s, "temporary", False):
                s.data = np.array(s.data, copy=True)
                s.temporary = True
            if begin is None:
                if s.data.shape != incoming.shape: s.data = np.zeros(incoming.shape, dtype=s.dtype)
                s.data[...] = incoming.astype(s.data.dtype, copy=False)
            else:
                b = tuple(int(x) for x in np.asarray(begin).ravel())
                e = tuple(int(x) for x in np.asarray(end).ravel())
                sl = tuple(slice(max(0,x), min(s.data.shape[i], e[i]+1)) for i,x in enumerate(b))
                s.data[sl] = incoming.astype(s.data.dtype, copy=False)
        self._dirty = True
        return True

    def lock_segments(self):
        """Apply the historical KDMS physical-layer lock.

        Source: segment.c::_kdms_lock_segment(). Non-permanent segments are
        destroyed during the lock; permanent segments become physically locked.
        The operation is intentionally one-way, matching the recovered API.
        """
        for name in list(self.segments):
            s=self.segments[name]
            if not s.permanence:
                del self.segments[name]
            else:
                s.locked=True
        return True

    def lock(self):
        self.lock_segments()
        return True

    def protect(self): self._protected = True; return self
    def close(self):
        # The historical close path flushes writable temporary state before the
        # final physical object is released.  The Python object currently keeps
        # its backing store resident, so closing records lifecycle state without
        # destroying data.
        self.closed = True
        return True
