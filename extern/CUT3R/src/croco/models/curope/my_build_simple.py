# build_simple.py
import torch
from torch.utils.cpp_extension import load
import os
import shutil



os.system('rm -rf /root/.cache/torch_extensions/py310_cu124/curope')

# 强制使用系统库路径
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LIBRARY_PATH', '')

print("强制使用系统库编译...")

# 清理旧文件
for f in os.listdir('.'):
    if f.endswith('.so'):
        os.remove(f)

print("开始 JIT 编译...")
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
        verbose=True,
        with_cuda=True
    )
    print("编译成功!")
    print("可用函数:", [x for x in dir(curope) if not x.startswith('_')])
    
except Exception as e:
    print(f"编译失败: {e}")
    import traceback
    traceback.print_exc()
