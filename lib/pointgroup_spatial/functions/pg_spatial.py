'''
Abastro, 2021
'''

import torch

import PG_SP

class Spatial:
    def __init__(self, hash_bits = 8):
        PG_SP.initialize()
        self.hash_bits = hash_bits
    def free(self):
        PG_SP.finalize()

    def cluster(self, radius, threshold, batch_idxs, coords, labels):
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda()
        assert coords.is_contiguous() and coords.is_cuda()
        assert labels.is_contiguous() and labels.is_cuda()

        with torch.no_grad():
            n = coords.size(0)
            cluster_idxs = labels.new()
            cluster_offsets = labels.new()
            PG_SP.clusterPoints(n, self.hash_bits, radius, threshold
                , batch_idxs, coords, labels, cluster_idxs, cluster_offsets)
            if cluster_idxs.size(0) == 0:
                print("Problem occurred")

        return cluster_idxs, cluster_offsets
