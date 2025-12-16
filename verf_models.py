# verify_models.py
import sys
import os

def find_models_directory():
    """查找 models 目录的实际位置"""
    print("=== 查找 models 目录 ===")
    
    search_paths = [
        "/home/jhc/projects/vmem/extern/CUT3R/src",
        "/home/jhc/projects/vmem/extern/CUT3R/src/croco", 
        "/home/jhc/projects/vmem/extern/CUT3R/src/croco/models",
        "/home/jhc/projects/vmem"
    ]
    
    for path in search_paths:
        models_path = os.path.join(path, "models")
        if os.path.exists(models_path):
            print(f"✅ 找到 models 目录: {models_path}")
            print(f"   目录内容: {os.listdir(models_path)[:5]}")  # 显示前5个文件
            
            # 检查是否有 curepe
            curepe_path = os.path.join(models_path, "curope")
            if os.path.exists(curepe_path):
                print(f"   ✅ 包含 curepe 目录: {os.listdir(curepe_path)[:5]}")
            else:
                print(f"   ❌ 不包含 curepe 目录")
                
            return path
    
    print("❌ 在所有路径中都没有找到 models 目录")
    return None

def test_import(models_base_path):
    """测试导入"""
    print(f"\n=== 测试导入 (基础路径: {models_base_path}) ===")
    
    # 添加基础路径到 sys.path
    if models_base_path not in sys.path:
        sys.path.insert(0, models_base_path)
        print(f"已添加路径: {models_base_path}")
    
    try:
        from models.curope import cuRoPE2D
        print("✅ 导入成功!")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

if __name__ == "__main__":
    models_base = find_models_directory()
    if models_base:
        test_import(models_base)
    
    # 显示当前 sys.path 中的 models 相关路径
    print(f"\n=== 当前 sys.path 中的 models 相关路径 ===")
    for path in sys.path:
        if "models" in path:
            print(f"  {path}")
