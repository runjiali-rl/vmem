import torch
import curope
import sys

# 将系统包目录插入到 Python 路径的最前面


def test_rope_2d_function():

    site_packages_path = '/home/jhc/apps/anaconda3/envs/vmem/lib/python3.10/site-packages'
    if site_packages_path in sys.path:
        sys.path.remove(site_packages_path)
        sys.path.insert(0, site_packages_path)

# 清理模块缓存
    for key in list(sys.modules.keys()):
        if 'curope' in key:
            del sys.modules[key]


    print(f"现在加载的模块: {curope.__file__}")
    print(f"模块属性: {[x for x in dir(curope) if not x.startswith('__')]}")

    if hasattr(curope, 'rope_2d'):
        print("✓ rope_2d 可用")
    else:
        print("✗ rope_2d 不可用")

    print("=== 测试 rope_2d 函数 ===")
    # 测试 CPU 版本
    print("\n1. 测试 CPU 版本:")
    B, N, H, D = 2, 5, 4, 16  # 较小的尺寸用于测试
    tokens_cpu = torch.randn(B, N, H, D * 4, device='cpu', dtype=torch.float32)
    positions_cpu = torch.randint(0, 10, (B, N, 2), dtype=torch.int64, device='cpu')
    
    print(f"CPU tokens shape: {tokens_cpu.shape}")
    print(f"CPU positions shape: {positions_cpu.shape}")
    
    try:
        # 保存原始值用于比较
        original_tokens_cpu = tokens_cpu.clone()
        
        # 调用函数
        result_cpu = curope.rope_2d(tokens_cpu, positions_cpu, 10000.0, 1.0)
        print("CPU 版本调用成功!")
        print(f"输出 shape: {result_cpu.shape}")
        print(f"输入和输出是同一个对象: {tokens_cpu is result_cpu}")
        print(f"数据是否被修改: {not torch.allclose(original_tokens_cpu, result_cpu)}")
    except Exception as e:
        print(f"CPU 版本调用失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试 CUDA 版本（如果有GPU）
    if torch.cuda.is_available():
        print("\n2. 测试 CUDA 版本:")
        tokens_cuda = torch.randn(B, N, H, D * 4, device='cuda', dtype=torch.float32)
        positions_cuda = torch.randint(0, 10, (B, N, 2), dtype=torch.int64, device='cuda')
        
        print(f"CUDA tokens shape: {tokens_cuda.shape}")
        print(f"CUDA positions shape: {positions_cuda.shape}")
        
        try:
            # 保存原始值用于比较
            original_tokens_cuda = tokens_cuda.clone()
            
            # 调用函数
            result_cuda = curope.rope_2d(tokens_cuda, positions_cuda, 10000.0, 1.0)
            print("CUDA 版本调用成功!")
            print(f"输出 shape: {result_cuda.shape}")
            print(f"输入和输出是同一个对象: {tokens_cuda is result_cuda}")
            print(f"数据是否被修改: {not torch.allclose(original_tokens_cuda, result_cuda)}")
        except Exception as e:
            print(f"CUDA 版本调用失败: {e}")
            import traceback
            traceback.print_exc()

test_rope_2d_function()
