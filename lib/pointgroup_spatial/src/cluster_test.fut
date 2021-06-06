-- Testing of cluster
-- Abastro, 2021
import "../lib/github.com/diku-dk/sorts/radix_sort"
import "cluster"

local let fst (x, _) = x
local let snd (_, y) = y

-- ==
-- entry: test_mkSegsFromSort_segInd
-- input { 4i64 [0, 0, 1, 1, 1, 3] [0, 0, 0, 0, 0, 0] }
-- output { [0i64, 2i64, 0i64, 5i64] }
-- input { 5i64 [0, 0, 1, 4] [1, 2, 2, 5] }
-- output { [0i64, 2i64, 0i64, 0i64, 3i64] }
entry test_mkSegsFromSort_segInd (numKey: i64) (keys: []i32) (values: []i32) : []i64 =
  (mkSegsFromSort numKey (keys, values)).segInd

-- ==
-- entry: test_mkSegsFromSort_segSize
-- input { 4i64 [0, 0, 1, 1, 1, 3] [0, 0, 0, 0, 0, 0] }
-- output { [2i64, 3i64, 0i64, 1i64] }
-- input { 5i64 [0, 0, 1, 4] [1, 2, 2, 5] }
-- output { [2i64, 1i64, 0i64, 0i64, 1i64] }
entry test_mkSegsFromSort_segSize (numKey: i64) (keys: []i32) (values: []i32) : []i64 =
  (mkSegsFromSort numKey (keys, values)).segSize


-- ==
-- entry: test_clusterWith
-- input { 1 6i64 [0, 1, 0, 4] [1, 2, 2, 5] }
-- output { [0, 1, 2, 4, 5] }
-- input { 3 9i64 [0, 1, 0, 4, 6, 7] [1, 2, 1, 5, 7, 8] }
-- output { empty([0]i32) }
-- input { 2 9i64 [0, 1, 0, 4, 6, 7] [1, 2, 1, 5, 7, 8] }
-- output { [0, 1, 2, 6, 7, 8] }
entry test_clusterWith (thres: i32) (numVert: i64) (graph_from: []i32) (graph_to: []i32) =
    let verts = iota numVert |> map i32.i64
    let sorted = (zip graph_from graph_to ++ zip verts verts)
        |> radix_sort_by_key fst i32.num_bits i32.get_bit
        |> unzip
    let graph = mkSegsFromSort numVert sorted
    in (clusterWith thres graph).raw

-- ==
-- entry: test_elimDupesWith
-- input { [1, 1, 1, 2, 3, 3, 5, 5, 5] }
-- output { [1, 2, 3, 5] }
entry test_elimDupesWith = elimDupesWith i32.to_i64

