# Modified version of PointGroup

Refer to [PointGroup](https://github.com/dvlab-research/PointGroup) for the basic setup.

## Installation

### Requirements
* Python 3.7.0
* Pytorch 1.8.1
* CUDA 10.2
* Futhark (0.20.0)
Futhark can be installed from [futhark homepage](futhark-lang.org/).

### Futhark Setup
First, do not forget to set up the following environment variables:
```
LIBRARY_PATH=/usr/local/cuda/lib64
LD_LIBRARY_PATH=/usr/local/cuda/lib64/
CPATH=/usr/local/cuda/include
```
Run the following to compile the futhark code into cuda-c.
```
cd lib/pointgroup_spatial
futhark cuda src/cluster.fut
```
Then, run
`CFLAGS=-fopenmp python setup.py develop`
to develop.
