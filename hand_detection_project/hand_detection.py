import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        初始化手部检测器
        
        Args:
            max_num_hands: 最大检测手的数量
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        # 手部关键点索引定义
        self.finger_tips = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小指指尖
        self.finger_pips = [3, 6, 10, 14, 18]  # 各手指PIP关节
        self.finger_mcp = [2, 5, 9, 13, 17]   # 各手指MCP关节
        self.wrist = 0                         # 手腕
        
        # 手指骨骼连接定义
        self.finger_connections = [
            # 拇指
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            # 食指
            [(0, 5), (5, 6), (6, 7), (7, 8)],
            # 中指
            [(0, 9), (9, 10), (10, 11), (11, 12)],
            # 无名指
            [(0, 13), (13, 14), (14, 15), (15, 16)],
            # 小指
            [(0, 17), (17, 18), (18, 19), (19, 20)]
        ]
        
        # 定义颜色方案
        self.colors = {
            'thumb': (255, 0, 0),      # 红色 - 拇指
            'index': (0, 255, 0),      # 绿色 - 食指
            'middle': (0, 0, 255),     # 蓝色 - 中指
            'ring': (255, 255, 0),     # 青色 - 无名指
            'pinky': (255, 0, 255),    # 紫色 - 小指
            'palm': (255, 255, 255),   # 白色 - 手掌
            'contour': (0, 255, 255)   # 黄色 - 轮廓
        }

    def detect_hands(self, image):
        """
        检测图像中的手部
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            results: MediaPipe手部检测结果
            rgb_image: RGB格式的图像
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        return results, rgb_image

    def draw_landmarks(self, image, results):
        """
        在图像上绘制手部关键点和连线
        
        Args:
            image: 输入图像
            results: MediaPipe检测结果
            
        Returns:
            image: 带有标记的图像
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                self.mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style()
                )
        return image

    def draw_enhanced_skeleton(self, image, results):
        """
        绘制增强的手部骨骼结构
        
        Args:
            image: 输入图像
            results: MediaPipe检测结果
            
        Returns:
            image: 带有骨骼结构的图像
        """
        if not results.multi_hand_landmarks:
            return image
            
        height, width, _ = image.shape
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for hand_landmarks in results.multi_hand_landmarks:
            # 获取所有关键点坐标
            landmarks = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmarks.append((x, y))
            
            # 绘制每个手指的骨骼
            for i, connections in enumerate(self.finger_connections):
                color = self.colors[finger_names[i]]
                for start_idx, end_idx in connections:
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    
                    # 绘制骨骼连线（粗线）
                    cv2.line(image, start_point, end_point, color, 4)
                    
                    # 绘制关节点
                    cv2.circle(image, start_point, 6, color, -1)
                    cv2.circle(image, end_point, 6, color, -1)
                    
                    # 绘制关节点边框
                    cv2.circle(image, start_point, 6, (0, 0, 0), 2)
                    cv2.circle(image, end_point, 6, (0, 0, 0), 2)
            
            # 额外绘制手掌连接线
            palm_connections = [(0, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
            for start_idx, end_idx in palm_connections:
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                cv2.line(image, start_point, end_point, self.colors['palm'], 3)
                
        return image

    def get_hand_contour(self, image, results):
        """
        获取优化的手部轮廓
        
        Args:
            image: 输入图像
            results: MediaPipe检测结果
            
        Returns:
            image: 带有轮廓的图像
        """
        height, width, _ = image.shape
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 获取所有关键点坐标
                points = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    points.append([x, y])
                
                # 转换为numpy数组
                points = np.array(points, dtype=np.int32)
                
                # 计算凸包来获得手部轮廓
                hull = cv2.convexHull(points)
                
                # 绘制多层轮廓效果
                # 外层轮廓（粗线）
                cv2.drawContours(image, [hull], -1, self.colors['contour'], 4)
                # 内层轮廓（细线）
                cv2.drawContours(image, [hull], -1, (255, 255, 255), 2)
                
                # 半透明填充
                overlay = image.copy()
                cv2.fillPoly(overlay, [hull], (*self.colors['contour'], 50))
                cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
                
        return image

    def get_finger_status(self, results):
        """
        分析手指状态（伸展/弯曲）
        
        Args:
            results: MediaPipe检测结果
            
        Returns:
            finger_status: 手指状态信息列表
        """
        finger_status = []
        
        if not results.multi_hand_landmarks:
            return finger_status
            
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y])
            
            fingers = []
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            
            # 拇指检测（横向比较）
            if landmarks[4][0] > landmarks[3][0]:  # 右手
                fingers.append(1 if landmarks[4][0] > landmarks[3][0] else 0)
            else:  # 左手
                fingers.append(1 if landmarks[4][0] < landmarks[3][0] else 0)
            
            # 其他四指检测（纵向比较）
            for i in range(1, 5):
                tip_idx = self.finger_tips[i]
                pip_idx = self.finger_pips[i]
                fingers.append(1 if landmarks[tip_idx][1] < landmarks[pip_idx][1] else 0)
            
            status = {
                'fingers': fingers,
                'finger_names': finger_names,
                'extended_count': sum(fingers),
                'gesture': self.recognize_gesture(fingers)
            }
            finger_status.append(status)
            
        return finger_status

    def recognize_gesture(self, fingers):
        """
        识别手势
        
        Args:
            fingers: 手指状态列表 [拇指, 食指, 中指, 无名指, 小指]
            
        Returns:
            gesture: 识别的手势名称
        """
        if fingers == [0, 0, 0, 0, 0]:
            return "拳头"
        elif fingers == [1, 1, 1, 1, 1]:
            return "手掌张开"
        elif fingers == [1, 1, 0, 0, 0]:
            return "胜利/数字2"
        elif fingers == [0, 1, 0, 0, 0]:
            return "指向"
        elif fingers == [0, 1, 1, 0, 0]:
            return "数字2"
        elif fingers == [0, 1, 1, 1, 0]:
            return "数字3"
        elif fingers == [0, 1, 1, 1, 1]:
            return "数字4"
        elif fingers == [1, 0, 0, 0, 0]:
            return "拇指向上"
        elif fingers == [1, 0, 0, 0, 1]:
            return "摇滚手势"
        else:
            return "其他手势"

    def draw_finger_analysis(self, image, finger_status):
        """
        在图像上绘制手指分析信息
        
        Args:
            image: 输入图像
            finger_status: 手指状态信息
            
        Returns:
            image: 带有分析信息的图像
        """
        if not finger_status:
            return image
            
        y_offset = 30
        for i, status in enumerate(finger_status):
            # 显示手指状态
            fingers_text = "Fingers: " + "".join([str(f) for f in status['fingers']])
            cv2.putText(image, fingers_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
            # 显示识别的手势
            gesture_text = f"Gesture: {status['gesture']}"
            cv2.putText(image, gesture_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
            
            # 显示伸展的手指数量
            count_text = f"Extended: {status['extended_count']}/5"
            cv2.putText(image, count_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 35
            
        return image

    def get_hand_info(self, results):
        """
        获取手部信息
        
        Args:
            results: MediaPipe检测结果
            
        Returns:
            hand_info: 包含手部信息的列表
        """
        hand_info = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # 获取手部类型（左手或右手）
                hand_type = handedness.classification[0].label
                confidence = handedness.classification[0].score
                
                # 获取关键点数量
                num_landmarks = len(hand_landmarks.landmark)
                
                info = {
                    'hand_index': i,
                    'hand_type': hand_type,
                    'confidence': confidence,
                    'num_landmarks': num_landmarks
                }
                hand_info.append(info)
        
        return hand_info

def main():
    """
    主函数 - 运行手部检测程序
    """
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 初始化手部检测器
    detector = HandDetector()
    
    print("手部轮廓检测已启动...")
    print("按 'q' 退出程序")
    print("按 's' 保存当前帧")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("错误：无法读取摄像头帧")
            break
        
        # 水平翻转图像（镜像效果）
        frame = cv2.flip(frame, 1)
        
        # 检测手部
        results, rgb_frame = detector.detect_hands(frame)
        
        # 获取手指状态和手势识别
        finger_status = detector.get_finger_status(results)
        
        # 创建四个不同的显示版本
        # 1. 原始图像 + 关键点
        landmarks_frame = frame.copy()
        landmarks_frame = detector.draw_landmarks(landmarks_frame, results)
        
        # 2. 增强骨骼结构
        skeleton_frame = frame.copy()
        skeleton_frame = detector.draw_enhanced_skeleton(skeleton_frame, results)
        
        # 3. 优化轮廓
        contour_frame = frame.copy()
        contour_frame = detector.get_hand_contour(contour_frame, results)
        
        # 4. 手势分析显示
        gesture_frame = frame.copy()
        gesture_frame = detector.draw_enhanced_skeleton(gesture_frame, results)
        gesture_frame = detector.draw_finger_analysis(gesture_frame, finger_status)
        
        # 获取手部信息
        hand_info = detector.get_hand_info(results)
        
        # 在手势分析帧上显示基本信息
        y_offset = 200  # 为手势信息留出空间
        cv2.putText(gesture_frame, f"Frame: {frame_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if hand_info:
            y_offset += 25
            cv2.putText(gesture_frame, f"Hands: {len(hand_info)}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            for info in hand_info:
                y_offset += 25
                text = f"{info['hand_type']}: {info['confidence']:.2f}"
                cv2.putText(gesture_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 创建2x2组合显示窗口
        height, width = frame.shape[:2]
        display_width = 320
        display_height = int(height * display_width / width)
        
        # 调整所有图像大小
        landmarks_small = cv2.resize(landmarks_frame, (display_width, display_height))
        skeleton_small = cv2.resize(skeleton_frame, (display_width, display_height))
        contour_small = cv2.resize(contour_frame, (display_width, display_height))
        gesture_small = cv2.resize(gesture_frame, (display_width, display_height))
        
        # 创建2x2网格
        top_row = np.hstack([landmarks_small, skeleton_small])
        bottom_row = np.hstack([contour_small, gesture_small])
        display_frame = np.vstack([top_row, bottom_row])
        
        # 添加标签
        cv2.putText(display_frame, "Landmarks", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, "Enhanced Skeleton", (display_width + 10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, "Contours", (10, display_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, "Gesture Analysis", (display_width + 10, display_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示结果
        cv2.imshow('Hand Detection - Press q to quit, s to save', display_frame)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存当前帧
            filename = f"hand_detection_frame_{frame_count}.jpg"
            cv2.imwrite(filename, gesture_frame)
            print(f"已保存帧: {filename}")
        
        frame_count += 1
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()
