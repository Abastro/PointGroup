'''
Abastro, 2021
'''

import torch

import PG_SP

class Spatial:
    def __init__(self, hash_bits = 8):
        PG_SP.initialize(0)
        self.hash_bits = hash_bits
    def free(self):
        PG_SP.finalize()

    def cluster(self, radius, threshold, batch_idxs, coords, labels):
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
        assert coords.is_contiguous() and coords.is_cuda
        assert labels.is_contiguous() and labels.is_cuda

        with torch.no_grad():
            n = coords.size(0)
            cluster_idxs = labels.new()
            cluster_offsets = labels.new()
            # print("Clustering")
            PG_SP.cluster_points(n, self.hash_bits, radius, threshold
                , batch_idxs.int(), coords, labels.int(), cluster_idxs.int(), cluster_offsets.int())
            # print("index over: {}".format(torch.nonzero(cluster_idxs[:][0]).view(-1).size()))
            # print("size: {}".format(cluster_offsets.size(0)))

        return cluster_idxs, cluster_offsets
