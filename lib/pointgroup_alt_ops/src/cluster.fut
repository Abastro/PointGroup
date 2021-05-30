-- Clustering algorithm based on voxel hashing and Hash-to-Min.
-- Abastro, 2021

import "../lib/github.com/diku-dk/segmented/segmented"
import "../lib/github.com/diku-dk/sorts/radix_sort"

local let fst (x, _) = x
local let snd (_, y) = y

-- | Finds flags out of the sorted array
local let segFlags [n] (arr: [n]i64) : [n]bool =
  map2 (!=) arr (rotate (-1) arr) with [0] = true

-- | Finds segment sizes from flags
local let flagToSizes [n] (flags: [n]bool) : []i64 =
  segmented_reduce (+) 0 flags (replicate n 1)

-- | Finds segment indices from segment sizes
local let sizeToIndices [n] (sizes: [n]i64) : *[n]i64 =
  let idxs = sizes |> scan (+) 0 |> rotate (-1)
  in copy idxs with [0] = 0

-- | Eliminates duplicates from the sorted array, with given representations for equality
local let elimDupesWith 'a (repr: a -> i64) (arr: []a) : []a =
  let flags = segFlags (map repr arr)
  let sizes = flagToSizes flags
  let idxs = sizeToIndices sizes
  in map (\i -> arr[i]) idxs


-- | Type representing index-based segments
type IndexSegs 'a [m][n] = {
  raw : [n]a
, segInd : [m]i64
, segSize : [m]i64 -- Invariant: interval btwn indices == sizes
}

-- | Expands with segments. Modified version of the expand function
local let expandSegs 'a 'b [m] (sz: a -> i64) (get: a -> i64 -> b) (arr: [m]a) : IndexSegs b [m][] =
  let szs = map sz arr
  let idxs = replicated_iota szs
  let flags = segFlags idxs
  let iotas = segmented_iota flags
  let segIdxs = sizeToIndices szs
  in { raw = map2 (\i j -> get arr[i] j) idxs iotas, segInd = segIdxs, segSize = szs }

-- | Constructs ranged key-based segments with the key-sorted array
local let mkSegsFromSort 'a [n] (numKey: i64) (keys: [n]i32, values: [n]a) : IndexSegs a [numKey][n] =
  let flags = segFlags (map i64.i32 keys)
  let segSizes = flagToSizes flags
  let segIdxs = sizeToIndices segSizes
  let segToKey = map (\i -> i64.i32 keys[i]) segIdxs
  let (keySizes, keyIdxs) =
    scatter (replicate numKey (0, 0)) segToKey (zip segSizes segIdxs) -- Missing key got index & size 0
    |> unzip
  in { raw = values, segInd = keyIdxs, segSize = keySizes }

-- | Entry of the segment
local let getEntry 'a (segs: IndexSegs a [] []) (s: i32) (i: i64) =
  segs.raw[segs.segInd[s] + i]


-- | Ball Query to query neighbors, along with itself.
-- nHashBit represents number of hash bits per coordinate
let ballQuery [n] (nHashBit: i32) (radius: f32)
  (pos: [n][3]f32) (labels: [n]i32) : IndexSegs i32 [n][] =
  let missing = -1
  let hashSize = 1 << (3 * nHashBit) |> i64.i32
  let hashMask = (1 << nHashBit) - 1
  let voxelOf v = (v / radius) |> f32.floor |> t32
  -- Subtract 0.5 to get i for searching close pts among (i), (i+1)
  let baseOf v = (v / radius) |> (+ (-0.5)) |> f32.floor |> t32
  let asHash vec =
    let lsb = map (& hashMask) vec
    in (lsb[2] << nHashBit | lsb[1]) << nHashBit | lsb[0]
  let asOff p = tabulate 3 (\i -> i64.get_bit (i32.i64 i) p)
  let isNeighbor p p' =
    let sqr r = r * r
    let sqRad = sqr radius
    let diff = map2 (-) pos[p] pos[p']
    in (reduce (+) 0 (map sqr diff) <= sqRad) && (labels[p] == labels[p'])

  -- Derivations
  let idxs = indices pos |> map i32.i64 -- [n]i32
  let vox = map (map voxelOf) pos -- [n][3]i32
  let base = map (map baseOf) pos -- [n][3]i32
  let vhs = map asHash vox -- (vHash)
  let vhPt = zip idxs vhs -- (pt, vHash)
  let sorted = vhPt |> radix_sort_by_key snd (3 * nHashBit) i32.get_bit -- Sorts on vHash
  let vHashToPt = mkSegsFromSort hashSize (unzip sorted) -- Segmented, vHash -> pt
  let maxSeg = i64.maximum vHashToPt.segSize

  -- Neighbor derivations
  let nbs pos = tabulate 8 asOff |> map (map2 (+) pos) -- denotes the neighbors, i32[8][3]
  let bringSeg s = tabulate maxSeg ( \i -> if i < vHashToPt.segSize[s]
    then getEntry vHashToPt s i else missing )
  let candidates = base -- candidates: [n][8][s]i32
    |> map (nbs >-> map asHash) -- calcualtes vHash of neighbors
    |> map (map bringSeg) -- queries correspondents

  let maxBatch = 8 * maxSeg
  let withSelf = maxBatch + 1
  -- sets those which does not satisfy predicate into missing
  let filterFn i j =
    if j == missing then missing
    else if isNeighbor i j then j else missing
  let found = candidates -- found: [n][8*s]i32
    |> map (flatten_to maxBatch)
    |> map2 (\i -> map (filterFn i)) idxs -- filters out non-neighbors
    |> map2 (\i arr -> (arr ++ [i]) :> [withSelf]i32) idxs -- Adds self
    |> map (radix_sort i32.num_bits i32.get_bit)  -- sort the results
  -- Note there is no duplicates as neighboring voxels attain different hashes
  -- Also -1 comes last in the sorted array
  let segSizes = found |> map (map (\j -> i32.bool (j != missing)) >-> i32.sum)
  in  expandSegs (\v -> i64.i32 segSizes[v]) (\v i -> found[v, i]) idxs


-- | Clustering with neighbors, using modified version of Hash-to-Min algorithm.
-- Only collects clusters bigger than threshold > 1.
let clusterWith [n] (thres: i32) (nbs: IndexSegs i32 [n][]) : IndexSegs i32 [][] =
  let idxs = indices nbs.segSize |> map i32.i64
  let invalidCls = { raw = [], segInd = replicate n 0, segSize = replicate n 0 }

  let updateStep (cls: IndexSegs i32 [n][]) : IndexSegs i32 [n][] =
    -- minimum in the v-cluster, v -> min(C_v)
    let minCls = map (\s -> cls.raw[s]) cls.segInd
    -- Distributes (target, received)
    let distributes = idxs |> expand
      ( \v -> cls.segSize[v] << 1 )
      ( \v i -> if (i & 1) == 0 -- Simple concatenation
        then (minCls[v], getEntry cls v (i >> 1))
        else (getEntry cls v (i >> 1), minCls[v]) )
    -- Sort by target, then by receive, then remove duplicatess
    let reprOf (t, r) = (i32.to_i64 t << i32.to_i64 i32.num_bits) | (i32.to_i64 r)
    let distributed = distributes
      |> radix_sort_by_key reprOf i64.num_bits i64.get_bit
      |> elimDupesWith reprOf
    in mkSegsFromSort n (unzip distributed) -- Turns it into segments

  let checkEqual cls newCls =
    let isEqSize = all id <| map2 (==) cls.segSize newCls.segSize
    in if isEqSize then all id <| map2 (==) cls.raw newCls.raw else false

  let (_, clusters) = loop (prevCls, cls) = (invalidCls, nbs)
    while !(checkEqual prevCls cls)
    do (cls, updateStep cls)

  -- Filtering here, only called once & it is convenient nonetheless
  let (toPick, _) = zip idxs clusters.segSize
    |> filter (\(_, sz) -> sz > i64.i32 thres)
    |> unzip
  in expandSegs (\v -> clusters.segSize[v]) (getEntry clusters) toPick

-- TODO Ignore stuff class. How?

-- TODO Piping stuff here
let clusterPoints [n] (nHashBit: i32) (radius: f32) (thres: i32)
  (pos: [n][3]f32) (labels: [n]i32) : IndexSegs i32 [][] =
  let nbs = ballQuery nHashBit radius pos labels
  in clusterWith thres nbs
