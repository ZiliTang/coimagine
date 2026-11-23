"""Lossless float32 row compression, resident RAM replay, unchanged indices."""
import ctypes as C
import hashlib
from pathlib import Path
import time

import numpy as np

from .dataset import read, save
from .full_data import Store, KEYS
from .long_online import PartitionedStore


class LZ4:
    def __init__(self):
        self.lib = C.CDLL('liblz4.so.1')
        self.lib.LZ4_compressBound.argtypes = [C.c_int]
        self.lib.LZ4_compressBound.restype = C.c_int
        self.lib.LZ4_compress_default.argtypes = [C.c_void_p,C.c_void_p,C.c_int,C.c_int]
        self.lib.LZ4_compress_default.restype = C.c_int
        self.lib.LZ4_decompress_safe.argtypes = [C.c_void_p,C.c_void_p,C.c_int,C.c_int]
        self.lib.LZ4_decompress_safe.restype = C.c_int
        self.lib.LZ4_versionString.restype = C.c_char_p
        self.version = self.lib.LZ4_versionString().decode()


def pack(source, destination):
    source, destination = Path(source), Path(destination)
    stat = (source/'x.npy').stat()
    identity = dict(path=str(source.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    if (destination/'manifest.json').exists():
        result = read(destination/'manifest.json')
        assert result['source'] == identity
        return result
    destination.mkdir(parents=True,exist_ok=True)
    x = np.load(source/'x.npy',mmap_mode='r')
    assert x.dtype == np.float32 and x.ndim == 2
    codec = LZ4()
    size = x.shape[1]*4
    capacity = codec.lib.LZ4_compressBound(size)
    buffer = C.create_string_buffer(capacity)
    restored = np.empty(x.shape[1],np.float32)
    offsets = np.zeros(len(x)+1,np.int64)
    raw_hash,blob_hash = hashlib.sha256(),hashlib.sha256()
    start = time.monotonic()
    with (destination/'x.lz4rows').open('wb') as stream:
        for i,row in enumerate(x):
            length = codec.lib.LZ4_compress_default(row.ctypes.data,buffer,size,capacity)
            if length <= 0:raise RuntimeError('LZ4 compression failed')
            written = codec.lib.LZ4_decompress_safe(buffer,restored.ctypes.data,length,size)
            if written != size or row.tobytes()!=restored.tobytes():
                raise ValueError('Lossless row roundtrip failed')
            encoded = buffer.raw[:length]
            stream.write(encoded)
            raw_hash.update(row.tobytes())
            blob_hash.update(encoded)
            offsets[i+1] = offsets[i]+length
            if (i+1)%50000 == 0:
                print('PACK_REPLAY',str(source),i+1,'/',len(x),flush=True)
    np.save(destination/'offsets.npy',offsets)
    result = dict(source=identity,shape=list(x.shape),dtype=str(x.dtype),codec='lz4-row',version=codec.version,
                  raw_sha256=raw_hash.hexdigest(),blob_sha256=blob_hash.hexdigest(),
                  raw_bytes=x.nbytes,packed_bytes=int(offsets[-1]),every_row_bitwise_verified=True,
                  seconds=time.monotonic()-start)
    save(destination/'manifest.json',result)
    return result


class ResidentStore:
    def __init__(self, source, packed):
        self.path = Path(source)
        self.metadata = read(self.path/'manifest.json')
        self.packed = pack(self.path,packed)
        self.width = self.packed['shape'][1]
        self.codec = LZ4()
        self.blob = np.fromfile(Path(packed)/'x.lz4rows',dtype=np.uint8)
        assert hashlib.sha256(self.blob).hexdigest() == self.packed['blob_sha256']
        self.offsets = np.load(Path(packed)/'offsets.npy')
        self.arrays = {k:np.load(self.path/(k+'.npy')) for k in (*KEYS,'next') if k!='x'}
        self.resident_bytes = self.blob.nbytes+self.offsets.nbytes+sum(v.nbytes for v in self.arrays.values())

    def __len__(self):return self.metadata['boundaries']

    def decode(self, ids):
        ids = np.asarray(ids)
        if np.any(ids<0) or np.any(ids>=len(self)):raise IndexError('replay index out of bounds')
        result = np.empty((len(ids),self.width),np.float32)
        size = self.width*4
        base = self.blob.ctypes.data
        dest = result.ctypes.data
        for i,index in enumerate(ids):
            start,end = self.offsets[index:index+2]
            written = self.codec.lib.LZ4_decompress_safe(base+int(start),dest+i*size,int(end-start),size)
            if written != size:raise ValueError('corrupt packed replay row')
        return result

    def batch(self, ids, next_state=False):
        if next_state:ids = self.arrays['next'][ids]
        return dict(x=self.decode(ids),**{k:np.array(self.arrays[k][ids],copy=True)
                    for k in ('candidates','sources','valid')})


class ReplayCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.stores = {}

    def get(self, path):
        path = Path(path).resolve()
        if path not in self.stores:
            name = hashlib.sha256(str(path).encode()).hexdigest()[:20]
            self.stores[path] = ResidentStore(path,self.directory/name)
        return self.stores[path]

    def online(self, paths):
        result = PartitionedStore.__new__(PartitionedStore)
        result.stores = [self.get(p) for p in paths]
        result.ends = np.cumsum([len(s) for s in result.stores])
        result.starts = np.r_[0,result.ends[:-1]]
        result.arrays = {k:np.concatenate([s.arrays[k] for s in result.stores])
                         for k in ('reward','done','chosen')}
        return result

    @property
    def resident_bytes(self):
        return sum(s.resident_bytes for s in self.stores.values())
