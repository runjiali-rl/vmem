# partial_static_build.py
import torch
from torch.utils.cpp_extension import load
import os

# 清理缓存
os.system('rm -rf /root/.cache/torch_extensions/py310_cu124/curope')

print("部分静态链接（只链接 stdc++）...")
try:
    curope = load(
        name='curope',
        sources=['curope.cpp', 'kernels.cu'],
        extra_cflags=['-O3', '-fPIC'],
        extra_cuda_cflags=[
            '-O3',
            '--ptxas-options=-v',
            '--use_fast_math',
        ],
        # 只静态链接 libstdc++，其他库动态链接
        extra_ldflags=[
            '-Wl,-Bstatic', '-lstdc++', '-Wl,-Bdynamic',
            '-lpthread',  # 确保 pthread 动态链接
        ],
        verbose=True,
        with_cuda=True
    )
    print("部分静态链接成功!")
    print(f"编译后的模块位置: {curope.__file__}") 
except Exception as e:
    print(f"编译失败: {e}")
    import traceback
    traceback.print_exc()
