# 详细检查函数信息
import inspect
import curope
import torch

def analyze_functions():
    for func_name in ['cuRoPE2D', 'curope2d']:
        if hasattr(curope, func_name):
            func = getattr(curope, func_name)
            print(f"\n=== {func_name} ===")
            print(f"文档: {func.__doc__}")
            try:
                print(f"签名: {inspect.signature(func)}")
            except:
                print("无法获取签名")
            
            # 尝试调用
            B, N, H, D = 2, 5, 4, 64  # 较小的尺寸用于测试
            tokens = torch.randn(B, N, H, D, device='cuda', dtype=torch.float32)
            positions = torch.randint(0, 10, (B, N, 2), dtype=torch.int64, device='cuda')
            
            try:
                result = func(tokens, positions, 10000.0, 1.0)
                print(f"调用成功! 输入: {tokens.shape}, 输出: {result.shape}")
            except Exception as e:
                print(f"调用失败: {e}")

analyze_functions()# 详细检查函数信息

