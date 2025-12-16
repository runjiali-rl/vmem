import os
import sys
import torch

# 1. 从sys.modules中彻底移除curope模块
modules_to_remove = [name for name in sys.modules if 'curope' in name]
for name in modules_to_remove:
    del sys.modules[name]

# 2. 清理所有可能的缓存目录
cache_paths = [
    '/root/.cache/torch_extensions/py310_cu124/curope',
    '/tmp/torch_extensions/py310_cu124/curope',
    './build',
    '/root/.cache/torch_extensions/curope'
]

for path in cache_paths:
    if os.path.exists(path):
        os.system(f'rm -rf "{path}"')
        print(f"已清理: {path}")

# 3. 强制垃圾回收
import gc
gc.collect()

print("缓存清理完成，重新编译模块...")

# 4. 重新编译
from torch.utils.cpp_extension import load

curope = load(
    name='curope',
    sources=['curope.cpp', 'kernels.cu'],
    extra_cflags=['-O3', '-fPIC'],
    extra_cuda_cflags=['-O3', '--ptxas-options=-v', '--use_fast_math'],
    extra_ldflags=['-Wl,-Bstatic', '-lstdc++', '-Wl,-Bdynamic', '-lpthread'],
    verbose=True
)

# 5. 立即检查
print("重新编译后的模块内容:", [x for x in dir(curope) if not x.startswith('__')])
print("模块文件:", getattr(curope, '__file__', '未知'))

# 6. 测试函数是否存在
if hasattr(curope, 'rope_2d'):
    print("✓ 成功找到 rope_2d 函数!")
else:
    print("✗ 仍然没有找到 rope_2d 函数")


# 立即验证并测试
print("编译后立即验证:")
print("模块内容:", [x for x in dir(curope) if not x.startswith('__')])

if hasattr(curope, 'rope_2d'):
    print("✓ 找到 rope_2d，立即测试...")

    # 立即测试
    B, N, H, D = 1, 2, 1, 16
    tokens = torch.randn(B, N, H, D * 4)
    positions = torch.randint(0, 10, (B, N, 2), dtype=torch.int64)
    original_tokens = tokens.clone()

    print(f"原始 tokens 形状: {tokens.shape}")
    print(f"原始 tokens 均值: {tokens.mean().item():.6f}")

    try:
        result = curope.rope_2d(tokens, positions, 10000.0, 1.0)
        print(result)
        # 检查输入张量是否被修改
        print(f"修改后 tokens 均值: {tokens.mean().item():.6f}")
        print(f"张量是否被修改: {not torch.allclose(original_tokens, tokens)}")
        print(f"修改量均值: {(tokens - original_tokens).mean().item():.6f}")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
else:
    print("✗ 没有找到 rope_2d")