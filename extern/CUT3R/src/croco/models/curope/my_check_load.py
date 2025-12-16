import sys
import os

# 强制清理所有 curepe 相关模块
for key in list(sys.modules.keys()):
    if 'curope' in key:
        print(f"删除模块: {key}")
        del sys.modules[key]

# 重新导入
import curope

print(f"实际加载的模块文件: {curope.__file__}")
print(f"模块所有属性: {[x for x in dir(curope) if not x.startswith('__')]}")

# 检查文件是否真的是我们替换的文件
expected_file = '/home/jhc/apps/anaconda3/envs/vmem/lib/python3.10/site-packages/curope.cpython-310-x86_64-linux-gnu.so'
if curope.__file__ == expected_file:
    print("✓ 加载的是替换后的文件")
else:
    print(f"✗ 加载的不是替换后的文件，而是: {curope.__file__}")


import hashlib

def compare_files():
    """比较两个文件的哈希值，确认替换是否成功"""

    original = '/root/.cache/torch_extensions/py310_cu124/curope/curope.so'
    replaced = '/home/jhc/apps/anaconda3/envs/vmem/lib/python3.10/site-packages/curope.cpython-310-x86_64-linux-gnu.so'

    def file_hash(filepath):
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    try:
        hash_original = file_hash(original)
        hash_replaced = file_hash(replaced)

        print(f"原始文件哈希: {hash_original}")
        print(f"替换文件哈希: {hash_replaced}")

        if hash_original == hash_replaced:
            print("✓ 文件替换成功，内容相同")
        else:
            print("✗ 文件内容不同，替换可能失败")

    except Exception as e:
        print(f"比较文件失败: {e}")

compare_files()