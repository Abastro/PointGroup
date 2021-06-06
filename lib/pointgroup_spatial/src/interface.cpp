// Intefacing for spatial operations: clustering
// Abastro, 2021
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include "cluster.h"

/// Interfaces
void initialize();
void finalize();
void clusterPoints (
    const int N
    , const int32_t nHashBit, const float radius, const int32_t thres
    , at::Tensor batch_idxs, at::Tensor pos, at::Tensor labels
    , at::Tensor cluster_idxs, at::Tensor cluster_offsets );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("initialize", &initialize, "initialize");
    m.def("finalize", &finalize, "finalize");
    m.def("cluster_points", &clusterPoints, "cluster_points");
}


/// Implementations

static struct futhark_context *ctxt;

void initialize()
{
    struct futhark_context_config *config = futhark_context_config_new();
    ctxt = futhark_context_new(config);
    futhark_context_config_free(config);
}

void finalize()
{
    futhark_context_free(ctxt);
}

/*
entry clusterPoints [n] (nHashBit: i32) (radius: f32) (thres: i32)
  (batches: [n]i32) (pos: [n][3]f32) (labels: [n]i32) : ([2][]i32, []i32)
In C:
int futhark_entry_clusterPoints(struct futhark_context *ctx,
                                struct futhark_i32_2d **out0,
                                struct futhark_i32_1d **out1, const int32_t in0,
                                const float in1, const int32_t in2, const
                                struct futhark_i32_1d *in3, const
                                struct futhark_f32_2d *in4, const
                                struct futhark_i32_1d *in5);
*/
void clusterPoints (
    const int N // Number of points
    , const int32_t nHashBit, const float radius, const int32_t thres
    , at::Tensor batch_idxs, at::Tensor pos, at::Tensor labels
    , at::Tensor cluster_idxs, at::Tensor cluster_offsets )
{
    // Note: Both is row-major
    struct futhark_i32_2d **fut_idxs = (struct futhark_i32_2d **) malloc(sizeof(void *));
    struct futhark_i32_1d **fut_offsets = (struct futhark_i32_1d **) malloc(sizeof(void *));
    int32_t *raw_batches = batch_idxs.data_ptr<int32_t>();
    float *raw_pos = pos.data_ptr<float>();
    int32_t *raw_labels = labels.data_ptr<int32_t>(); // Need to check row-major, but.. laziness
    const struct futhark_i32_1d *fut_batches = futhark_new_i32_1d(ctxt, raw_batches, N);
    const struct futhark_f32_2d *fut_pos = futhark_new_f32_2d(ctxt, raw_pos, N, 3);
    const struct futhark_i32_1d *fut_labels = futhark_new_i32_1d(ctxt, raw_labels, N);

    // This function also automatically allocates the output ptr
    futhark_entry_clusterPoints(ctxt, fut_idxs, fut_offsets
        , nHashBit, radius, thres, fut_batches, fut_pos, fut_labels);

    int nActive = (int) futhark_shape_i32_2d(ctxt, *fut_idxs)[1];
    int nCluster = (int) futhark_shape_i32_1d(ctxt, *fut_offsets)[0];
    cluster_idxs.resize_({nActive, 2});
    cluster_offsets.resize_({nCluster + 1});
    futhark_values_i32_2d(ctxt, *fut_idxs, cluster_idxs.data_ptr<int32_t>());
    futhark_values_i32_1d(ctxt, *fut_offsets, cluster_offsets.data_ptr<int32_t>());

    futhark_free_i32_2d(ctxt, *fut_idxs);
    futhark_free_i32_1d(ctxt, *fut_offsets);
    free(fut_idxs);
    free(fut_offsets);
}
