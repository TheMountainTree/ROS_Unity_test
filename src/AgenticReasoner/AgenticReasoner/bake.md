# 节点功能工具

节点工具是指在部分包中提供的功能性工具，通常用于特定任务，如图像处理、路径规划等。

## 图像获取工具
- 包路径: `/home/frank/workspace/Picture_Capture`
- 功能：使用orbbec相机获取场景图像
- 使用流程： 打开包路径后，执行`source install/setup.bash && ros2 run get_picture SingleCamera1SaveNode`，即可启动相机捕获程序。随后在新的终端中执行`ros2 param set /single_camera1_save_node save_trigger true`，即可触发相机捕获图像并保存到路径`/home/frank/workspace/Picture_Capture/data/camera1`中，以相机`名称_图像类型_时间戳.格式`命名。这里的节点获取的是`camera1_color_{时间戳}.png`
- 注意事项：相机程序是持续节点，需要手动停止。