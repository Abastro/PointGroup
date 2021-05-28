-- Uses diku-dk's radix-sort
import "../lib/github.com/diku-dk/sorts/radix_sort"

local let fst (x, y) = x
local let snd (x, y) = y
local let gather 'a (xs: []a) (is: []i32) =
  map (\i -> xs[i]) is

-- | Finds changed indices in a sorted array
local let changedIndices [n] (sorted: [n]i32): []i32 =
  let inds = indices sorted
  let change = map2 (!=) (rotate 1 sorted) sorted
  -- MAYBE Filter is expensive. Alternative ways?
  let midChInds = zip change inds |> init |> filter fst |> map snd
  in 0 ++ midChInds n ++ n

-- | Ball Query.
-- nHashBit represents number of hash bits per coordinate
entry ballQuery [n] (nHashBit: i32) (radius: f32) (pos: [n][3]f32) =
  let missing = -1
  let hashUnit = 1 << nHashBit
  let hashSize = 1 << (3 * nHashBit)
  let hashMask = hashUnit - 1
  let voxelOf v = f32.floor (v / radius) |> t32
  -- Subtract 0.5 to get i for searching close pts among (i), (i+1)
  let baseOf v = f32.floor (v / radius) |> (+ (-0.5)) |> t32
  let asHash vec =
    let lsb = map (& hashMask) vec
    in (lsb[0] * hashUnit + lsb[1]) * hashUnit + lsb[2]
  let asOff p = tabulate 3 (\i -> i32.get_bit i p)

  -- Derivations
  let inds = indices pos
  let vox = map (map voxelOf) pos
  let base = map (map searchBaseOf) pos
  let vhs = map asHash vox -- (vHash)
  let vhPt = zip inds vhs -- (pt, vHash)
  -- Sorts based on vHash
  let sorted = radix_sort_by_key snd (3 * nHashBit) i32.get_bit vhPt
  let segAll = changedIndices (map snd sorted) -- vHash-group segment indices
  let nVoxel = length segs - 1
  let segs = take nVoxel segAll
  -- Maximum number of vertices in voxel
  let maxInVox = map2 (-) (rotate 1 segAll) segAll |> init |> i32.maximum
  let segEntry s i = segs[s] + i
  -- TODO Invalidate entries? There may be too many entries
  let segPts = tabulate_2d nVoxel maxInVoxel (\s i -> sorted[segEntry s i])
  let segTovh = gather vhs segs
  let vhToSeg = scatter (replicate hashSize missing) segTovh inds
  let candidate off = base
    |> map (map (+) off >-> asHash) -- adds offset & calc hash
    |> flip gather vhToSeg -- converts to segment index (or -1 if missing)
    -- maps to the corresponding voxel segment pts
    |> map (\s -> if s == missing then replicate maxInVox missing else segPts[s])
  let candidates = tabulate 8 (asOff >-> candidate)
  -- TODO Filter all here with requirement

  in undefined

  sizeSeg s = A.cond (s A.== missing) 0 $ segs A.!! (s+1) - segs A.!! s
  entrySeg i s = A.fst $ sorted A.!! (segs A.!! s + i) -- Pt # for the index

  inRadius p p' = let
    sqr r = r * r
    A.T3 x y z = posT A.!! p; A.T3 x' y' z' = posT A.!! p'
    in sqr (x' - x) + sqr (y' - y) + sqr (z' - z) A.<= A.lift (sqr radius)


