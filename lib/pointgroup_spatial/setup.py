from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='PG_SP',
    ext_modules=[
        CUDAExtension('PG_SP', [
            'src/cluster.c'
            , 'src/interface.cpp'
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
