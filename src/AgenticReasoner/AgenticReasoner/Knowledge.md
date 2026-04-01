# 背景知识库

本文档包含Agent运行所需的背景知识，可在系统提示词之外提供额外的上下文信息。

# 环境信息
- 操作系统: Ubuntu Linux 22.04
- ROS版本: ROS2 
- Shell: Bash

## 环境工具
 - ubuntu 命令行工具: "tree"
    - 用于查看目录结构
    - 示例: `tree -L 2`


# Skills
## 环境感知
### camera_tools_knowledge:
- 功能：提供与相机相关的工具和操作流程，帮助Agent获取当前场景的图像信息，需要把注意力放在金属桌面区域，忽略其他区域。环境感知后需要结合两张相机输出的图像，判断桌面上有哪些物体，他们的位置、大小、颜色等信息。
- 工作路径：`/home/frank/workspace/Picture_Capture`
- 输出目录：`/home/frank/workspace/Picture_Capture/data/camera1`和`/home/frank/workspace/Picture_Capture/data/camera2`
- 使用注意：
  - 默认情况下请使用双相机感知环境。每个相机都需要在一个独立的子进程中运行。但捕获图像时，请同时使用两个相机的触发信号保证图像的同步性。

- 节点工具
  - SingleCamera1SaveNode
    - 描述：Orbbec 相机1的底层驱动与控制节点。必须使用【异步执行工具】启动并保持运行。运行期间会独占相机硬件。
    - 类型: "continuous_background_service"
    - 启动命令: `source install/setup.bash && ros2 run get_picture SingleCamera1SaveNode`
    - 节点使用: `ros2 param set /single_camera1_save_node save_trigger true`向相机节点发送信号，触发一次图像捕获并保存到硬盘。前提是 SingleCamera1SaveNode 必须处于运行状态。

  - SingleCamera2SaveNode
    - 描述：Orbbec 相机2的底层驱动与控制节点。必须使用【异步执行工具】启动并保持运行。运行期间会独占相机硬件。
    - 类型: "continuous_background_service"
    - 启动命令: `source install/setup.bash && ros2 run get_picture SingleCamera2SaveNode`
    - 节点使用: `ros2 param set /single_camera2_save_node save_trigger true`向相机节点发送信号，触发一次图像捕获并保存到硬盘。前提是 SingleCamera2SaveNode 必须处于运行状态。

### GroundedSAM2_keypoints_recognize_knowledge:
- 功能: 使用GroundedSAM2基于vlm输出的场景描述json进行实例识别和分割，并基于分割结果使用dinov2进行关键点提取。最终可以获得关键点的描述文件。
- 工作路径: `/home/frank/workspace/eeg_robot`
- 输出目录: `/home/frank/workspace/eeg_robot/results`
- 使用注意:
  - 需要先打开工作路径后再运行文件
  - 需要激活环境`conda activate mani_eeg`，然后进入工作路径`/home/frank/workspace/eeg_robot/src/robot_ctr/graph/graph`，运行`python dual_view_pipeline_service_with_pcd_save4.py`
  - 程序启动后，需要设置参数开始运行`ros2 topic pub /task_state std_msgs/msg/String "{data: 'B'}" --once`，触发一次图像处理和关键点提取流程。前提是程序必须处于运行状态。
  - 每次运行都必须从状态A转换到状态B。


# 常见问题处理

## Topic不存在
如果提示topic不存在，通常需要先启动对应的节点：
1. 检查可用节点: `ros2 node list`
2. 检查可用话题: `ros2 topic list`
3. 启动必要节点后再尝试操作

## 权限问题
某些操作可能需要sudo权限，但应谨慎使用。

## 进程管理
- 后台运行的进程会返回PID
- 可以使用 `<stop_background pid="PID">` 停止进程
- 使用 `<list_background>` 查看所有后台进程