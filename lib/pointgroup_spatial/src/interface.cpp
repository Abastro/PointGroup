// Intefacing for spatial operations: clustering
// Abastro, 2021
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include "cluster.h"

/// Interfaces
void initialize(const int profile);
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

static int do_profile;
static struct futhark_context *ctxt;

void initialize(const int profile){
    struct futhark_context_config *config = futhark_context_config_new();
    if(profile) {
        futhark_context_config_set_profiling(config, 1);
        do_profile = profile;
    }
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
    struct futhark_i32_2d *fut_idxs;
    struct futhark_i32_1d *fut_offsets;
    int32_t *raw_batches = batch_idxs.data_ptr<int32_t>();
    float *raw_pos = pos.data_ptr<float>();
    int32_t *raw_labels = labels.data_ptr<int32_t>(); // Need to check row-major, but.. laziness
    struct futhark_i32_1d *fut_batches = futhark_new_i32_1d(ctxt, raw_batches, N);
    struct futhark_f32_2d *fut_pos = futhark_new_f32_2d(ctxt, raw_pos, N, 3);
    struct futhark_i32_1d *fut_labels = futhark_new_i32_1d(ctxt, raw_labels, N);

    // This function also automatically allocates the output ptr
    futhark_entry_clusterPoints(ctxt, &fut_idxs, &fut_offsets
        , nHashBit, radius, thres, fut_batches, fut_pos, fut_labels);

    int nActive = (int) futhark_shape_i32_2d(ctxt, fut_idxs)[0];
    int nClusterAnd1 = (int) futhark_shape_i32_1d(ctxt, fut_offsets)[0];
    cluster_idxs.resize_({nActive, 2});
    cluster_offsets.resize_({nClusterAnd1});
    int32_t *raw_idxs = cluster_idxs.data_ptr<int32_t>();
    int32_t *raw_offsets = cluster_offsets.data_ptr<int32_t>();
    futhark_values_i32_2d(ctxt, fut_idxs, raw_idxs);
    futhark_values_i32_1d(ctxt, fut_offsets, raw_offsets);

    futhark_free_i32_1d(ctxt, fut_batches);
    futhark_free_f32_2d(ctxt, fut_pos);
    futhark_free_i32_1d(ctxt, fut_labels);
    futhark_context_clear_caches(ctxt);
    if(do_profile)
        printf("Log: %s\n", futhark_context_report(ctxt));
}
