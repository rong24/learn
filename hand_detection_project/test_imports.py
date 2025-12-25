#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试手部检测代码的导入和基本功能
"""

import sys
import os

def test_imports():
    """测试所有必要的导入"""
    try:
        import cv2
        print("✓ OpenCV 导入成功")
        print(f"  版本: {cv2.__version__}")
        
        import mediapipe as mp
        print("✓ MediaPipe 导入成功")
        print(f"  版本: {mp.__version__}")
        
        import numpy as np
        print("✓ NumPy 导入成功")
        print(f"  版本: {np.__version__}")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_hand_detector():
    """测试HandDetector类的初始化"""
    try:
        # 导入HandDetector类
        sys.path.append(os.path.dirname(__file__))
        from hand_detection import HandDetector
        
        # 测试初始化
        detector = HandDetector()
        print("✓ HandDetector 初始化成功")
        
        # 测试MediaPipe组件
        if hasattr(detector, 'mp_hands'):
            print("✓ MediaPipe hands 模块加载成功")
        if hasattr(detector, 'hands'):
            print("✓ MediaPipe Hands 对象创建成功")
        if hasattr(detector, 'mp_draw'):
            print("✓ MediaPipe 绘图工具加载成功")
        if hasattr(detector, 'mp_draw_styles'):
            print("✓ MediaPipe 绘图样式加载成功")
            
        return True
    except Exception as e:
        print(f"✗ HandDetector 测试失败: {e}")
        return False

def test_image_processing():
    """测试图像处理功能"""
    try:
        import cv2
        import numpy as np
        sys.path.append(os.path.dirname(__file__))
        from hand_detection import HandDetector
        
        # 创建测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (100, 150, 200)  # 填充颜色
        
        detector = HandDetector()
        
        # 测试手部检测函数（虽然空图像不会检测到手部）
        results, rgb_image = detector.detect_hands(test_image)
        print("✓ 手部检测函数运行正常")
        
        # 测试绘制函数
        output_image = detector.draw_landmarks(test_image.copy(), results)
        print("✓ 关键点绘制函数运行正常")
        
        # 测试轮廓函数
        contour_image = detector.get_hand_contour(test_image.copy(), results)
        print("✓ 轮廓绘制函数运行正常")
        
        # 测试信息获取函数
        hand_info = detector.get_hand_info(results)
        print("✓ 手部信息获取函数运行正常")
        print(f"  检测到的手部数量: {len(hand_info)}")
        
        return True
    except Exception as e:
        print(f"✗ 图像处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("手部检测代码测试")
    print("=" * 50)
    
    # 测试导入
    print("\n1. 测试库导入:")
    imports_ok = test_imports()
    
    # 测试HandDetector类
    print("\n2. 测试HandDetector类:")
    detector_ok = test_hand_detector()
    
    # 测试图像处理
    print("\n3. 测试图像处理功能:")
    processing_ok = test_image_processing()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    if imports_ok:
        print("✓ 所有依赖库导入正常")
    else:
        print("✗ 依赖库导入有问题")
    
    if detector_ok:
        print("✓ HandDetector类工作正常")
    else:
        print("✗ HandDetector类有问题")
        
    if processing_ok:
        print("✓ 图像处理功能正常")
    else:
        print("✗ 图像处理功能有问题")
    
    if imports_ok and detector_ok and processing_ok:
        print("\n🎉 所有测试通过！代码已修复并可以正常运行。")
        print("\n要运行手部检测程序，请确保:")
        print("  1. 连接摄像头")
        print("  2. 运行: python hand_detection.py")
        print("  3. 将手放在摄像头前进行测试")
        return True
    else:
        print("\n❌ 部分测试失败，需要进一步调试。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
