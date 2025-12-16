import os
import torch
from torch.utils.cpp_extension import load

# 清理缓存
os.system('rm -rf /root/.cache/torch_extensions/py310_cu124/curope')

print("尝试只编译 curepe.cpp...")
try:
    curepe_simple = load(
        name='curope_simple',
        sources=['curope.cpp'],  # 只使用 cpp 文件
        extra_cflags=['-O3', '-fPIC'],
        extra_ldflags=['-Wl,-Bstatic', '-lstdc++', '-Wl,-Bdynamic', '-lpthread'],
        verbose=True
    )
    
    print("简单版本中的函数:", [x for x in dir(curope_simple) if not x.startswith('__')])
    
    # 测试 rope_2d 是否存在
    if hasattr(curope_simple, 'rope_2d'):
        print("找到 rope_2d 函数!")
        func = curepe_simple.rope_2d
        print(f"类型: {type(func)}")
        if callable(func):
            print(f"可调用，签名: {inspect.signature(func)}")
            
except Exception as e:
    print(f"单独编译失败: {e}")
