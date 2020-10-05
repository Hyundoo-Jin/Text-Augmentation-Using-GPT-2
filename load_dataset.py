import glob
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
    

def load_dataset(enc, path): 
    token_chunks = list()
    length = 0
    with open(path, 'r') as f :
        while True :
            line = f.readline()
            if not line : break
            length += 1
    with open(path, 'r') as fp :
        tokens = list()
        for _ in tqdm(range(length)) :
            line = fp.readline()
            tokens.extend(enc.EncodeAsIds(line))
            if '<|endoftext|>' in line :
                token_chunks.append(np.array(tokens))
                tokens = list()
    return token_chunks

def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi
    
class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.
    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks, seed=None, transfer = None, labels = False) :
        self.chunks = chunks
        self.labels = labels
        if self.labels :
            self.total_size = sum(len(chunk[0]) for chunk in chunks)
        else :
            self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        self.transfer = transfer
        for i in range(len(chunks)):
            if self.labels :
                self.boundaries.append(self.boundaries[-1] + len(chunks[i][0]))
            else :
                self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])
        self.rs = np.random.RandomState(seed=seed)


    def sample(self, length, generate = False, after = False):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time. (Under {} tokens needed.)".format(
            length, self.total_size // len(self.chunks))
        while True:
            index = self.rs.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                            len(self.boundaries) - 1) - 1
            if self.transfer :
                result = self.chunks[i]
                if generate :
                    if self.labels :
                        idx = np.where(result[0] == 2)[0][0]
                        if after :
                            return result[0][:idx], result[1], result[2]
                        else :
                            return result[0][:idx], result[1]
                    else :
                        idx = np.where(result == 2)[0][0]
                        return result[:idx]
                else :
                    return result
            else :
                if self.boundaries[i + 1] > index + length:
                    within_chunk = index - self.boundaries[i]
                    return self.chunks[i][within_chunk:within_chunk + length]

    def sample_simple(self, q_id, batch_size, generate = False, after = False):
        chunk = self.chunks_dict[q_id]
        index = np.random.choice(len(chunk), batch_size)
        result = chunk[index]
        return result
                
    def add_ids(self, q_ids) :
        self.chunks_dict = dict()
        a = 0
        for id in range(max(q_ids) + 1) :
            b = sum(q_ids == id)
            self.chunks_dict[id] = np.array(self.chunks[a:a+b])
            a += b
            print('Q_ID : {} added. samples : {}'.format(id, b))
        del self.chunks
        print('id 추가 완료')