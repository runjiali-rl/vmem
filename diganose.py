# diagnose_import.py
import sys
import os

print("=== Python 路径诊断 ===")
print(f"Python 版本: {sys.version}")
print(f"Python 可执行文件: {sys.executable}")

print("\n=== PYTHONPATH 环境变量 ===")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '未设置')}")

print("\n=== sys.path 内容 ===")
for i, path in enumerate(sys.path):
    print(f"{i:2d}: {path}")

print("\n=== 查找 models 模块 ===")
models_found = []
for path in sys.path:
    models_path = os.path.join(path, "models")
    if os.path.exists(models_path):
        models_found.append(models_path)
        print(f"✅ 找到: {models_path}")

if not models_found:
    print("❌ 未找到 models 目录")

print("\n=== 尝试导入 ===")
try:
    from models.curope import cuRoPE2D
    print("✅ 导入成功!")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    
    # 更详细的错误信息
    import traceback
    traceback.print_exc()
