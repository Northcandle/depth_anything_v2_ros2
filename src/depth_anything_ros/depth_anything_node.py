#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import sys
import os
from torchvision.transforms import Compose

# 设置 Depth Anything V2 路径
DEPTH_ANYTHING_PATH = os.path.expanduser("~/Depth-Anything-V2")
if os.path.exists(DEPTH_ANYTHING_PATH):
    sys.path.append(DEPTH_ANYTHING_PATH)
else:
    print(f"ERROR: Depth Anything V2 not found at {DEPTH_ANYTHING_PATH}")
    sys.exit(1)
# 导入 Depth Anything 模块
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_anything_v2.dpt import DepthAnythingV2

class DepthAnythingNode(Node):
    def __init__(self):
        super().__init__('depth_anything_node')
        
        # 参数配置
        self.declare_parameter('model_size', 'small')  # small/base/large
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('depth_topic', '/depth_map')
        self.declare_parameter('color_topic', '/depth_map_color')  # 新增伪彩色话题
        self.declare_parameter('device', 'cuda')
        
        model_size = self.get_parameter('model_size').value
        input_topic = self.get_parameter('input_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        color_topic = self.get_parameter('color_topic').value
        device = self.get_parameter('device').value
        
        # 加载模型
        self.get_logger().info("Loading Depth Anything V2 model...")
        self.model = self.load_model(model_size, device)
        self.transform = self.create_transform()
        
        # ROS 工具
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )
        self.depth_publisher = self.create_publisher(Image, depth_topic, 10)
        self.color_publisher = self.create_publisher(Image, color_topic, 10)  # 新增发布器
        self.get_logger().info(f"Depth Anything V2 node ready (using {model_size} model)")

    def load_model(self, size, device):
        # 修正模型文件名 (使用实际文件名格式)
        model_name = f'depth_anything_v2_vit{"s" if size == "small" else "b" if size == "base" else "l"}.pth'
        model_path = os.path.join(DEPTH_ANYTHING_PATH, 'checkpoints', model_name)
        
        self.get_logger().info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if size == 'small':
            model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        elif size == 'base':
            model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
        else:  # large
            model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
        
        model.load_state_dict(torch.load(model_path))
        model.to(device).eval()
        return model

    def create_transform(self):
        return Compose([
            Resize(
                width=518, 
                height=518, 
                resize_target=False,
                keep_aspect_ratio=True, 
                ensure_multiple_of=14,
                resize_method='lower_bound', 
                image_interpolation_method=cv2.INTER_CUBIC
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet()
        ])

    def image_callback(self, msg):
        try:
            # 转换ROS图像为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 预处理
            image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) / 255.0
            transformed = self.transform({'image': image})['image']
            tensor = torch.from_numpy(transformed).unsqueeze(0).to(self.get_parameter('device').value)
            
            # 深度估计
            with torch.no_grad():
                depth = self.model(tensor)
            depth = depth.squeeze().cpu().numpy()
            
            # 发布原始深度图 (32FC1格式)
            depth_raw = depth.astype(np.float32)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_raw, encoding='32FC1')
            depth_msg.header = msg.header
            self.depth_publisher.publish(depth_msg)
            
            # 创建伪彩色深度图 (用于可视化)
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_color = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # 发布伪彩色深度图
            color_msg = self.bridge.cv2_to_imgmsg(depth_color, encoding='bgr8')
            color_msg.header = msg.header
            self.color_publisher.publish(color_msg)
            
            # 调试信息
            self.get_logger().debug(f"Processed image: {msg.width}x{msg.height} -> Depth: {depth_raw.shape}")

        except Exception as e:
            self.get_logger().error(f"Processing error: {str(e)}", throttle_duration_sec=5)

def main(args=None):
    rclpy.init(args=args)
    node = DepthAnythingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()