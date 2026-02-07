"""
MTCNN版本诊断脚本
用于确定MTCNN的正确参数
"""

import sys

print("=" * 70)
print("MTCNN版本诊断")
print("=" * 70)

# 检查MTCNN是否安装
try:
    import mtcnn

    print(f"\n✅ MTCNN已安装")
    print(f"版本: {mtcnn.__version__ if hasattr(mtcnn, '__version__') else '未知'}")
except ImportError:
    print("\n❌ MTCNN未安装")
    print("安装命令: pip install mtcnn --break-system-packages")
    sys.exit(1)

# 检查MTCNN类
try:
    from mtcnn import MTCNN

    print(f"✅ MTCNN类导入成功")
except ImportError as e:
    print(f"❌ MTCNN类导入失败: {e}")
    sys.exit(1)

# 检查初始化方法
print(f"\n【检查 MTCNN.__init__ 方法】")

import inspect

# 获取__init__方法的签名
sig = inspect.signature(MTCNN.__init__)
print(f"\n完整签名:")
print(f"  MTCNN.__init__{sig}")

print(f"\n参数列表:")
for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue

    default = param.default
    default_str = f"={default}" if default != inspect.Parameter.empty else ""
    print(f"  - {param_name}{default_str}")

# 尝试不同的初始化方式
print(f"\n【尝试初始化】")

test_cases = [
    ("默认参数", {}),
    ("min_face_size", {"min_face_size": 40}),
    ("min_detection_size", {"min_detection_size": 40}),
    ("scale_factor", {"scale_factor": 0.709}),
    ("组合参数", {"min_face_size": 40, "scale_factor": 0.709}),
]

for name, kwargs in test_cases:
    try:
        detector = MTCNN(**kwargs)
        print(f"  ✅ {name}: {kwargs}")

        # 尝试检测
        import numpy as np

        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_faces(test_img)
        print(f"     检测方法可用: detect_faces()")

        break  # 成功则退出

    except TypeError as e:
        print(f"  ❌ {name}: {e}")
    except Exception as e:
        print(f"  ⚠️  {name}: {e}")

# 检查检测方法
print(f"\n【检查可用方法】")
methods = [m for m in dir(MTCNN) if not m.startswith('_')]
print(f"可用方法: {', '.join(methods)}")

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)

# 生成推荐配置
print(f"\n📋 推荐配置:")
print(f"""
# 初始化MTCNN检测器
from mtcnn import MTCNN

# 方式1: 无参数（最安全）
detector = MTCNN()

# 方式2: 根据上述测试结果选择可用参数
# detector = MTCNN(min_face_size=40)  # 如果支持
""")