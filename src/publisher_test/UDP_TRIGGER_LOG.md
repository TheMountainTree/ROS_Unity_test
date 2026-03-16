# UDP Trigger 通信系统日志

## 系统概述

该系统由两个组件构成，用于在 ROS2 环境与 Windows 串口设备之间传递 Trigger 信号：

| 组件 | 文件 | 运行环境 | 功能 |
|------|------|----------|------|
| UDP 发送端 | `publisher_test/udp_sender_node.py` | ROS2 节点 | 发送 Trigger 信号 |
| UDP 接收端 | `windows_com_publisher.py` | Windows | 接收并转发至串口 |

---

## 1. UDP 发送端 (`udp_sender_node.py`)

**功能描述：** ROS2 节点，按指定频率通过 UDP 协议发送 Trigger 信号。

### 参数配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `local_ip` | 192.168.56.103 | 本地绑定 IP |
| `local_port` | 5006 | 本地绑定端口 |
| `remote_ip` | 192.168.56.3 | 目标 IP |
| `remote_port` | 8888 | 目标端口 |
| `trigger_value` | 1 | 初始 Trigger 值 (0-255) |
| `auto_increment` | False | 是否自动递增 |
| `publish_hz` | 2.0 | 发送频率 (Hz) |

### 日志输出示例

```
[INFO] UDP trigger sender started: 192.168.56.103:5006 -> 192.168.56.3:8888, 2.00 Hz, trigger=1, auto_increment=False
[INFO] Sent trigger: 1 (HEX: 01)
```

---

## 2. UDP 接收端 (`windows_com_publisher.py`)

**功能描述：** Windows 端脚本，监听 UDP 端口，将接收到的 Trigger 信号通过 DCP 协议转发至串口。

### 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| UDP_IP | 192.168.56.3 | 监听 IP |
| UDP_PORT | 8888 | 监听端口 |
| COM_PORT | COM3 | 串口设备 |
| BAUD_RATE | 115200 | 波特率 |

### DCP 协议格式

```
[0x01, 0xE1, 0x01, 0x00, trigger_value]
```

### 日志输出示例

```
[*] 正在监听 UDP 端口 8888...
[*] 成功打开 COM 端口 COM3，波特率 115200
[*] 等待接受 Trigger 信号 ...
[1709423456.7890] 已转发 Trigger: 1 (HEX: 01)  来自 ('192.168.56.103', 5006)
```

---

## 3. 数据流

```
┌─────────────────────┐       UDP        ┌─────────────────────┐      Serial      ┌─────────────┐
│   ROS2 节点          │  ─────────────>  │   Windows 接收端     │  ─────────────>  │  串口设备    │
│  udp_sender_node    │   Trigger (1B)   │ windows_com_pub     │   DCP Command   │   (COM3)    │
│  192.168.56.103     │                  │  192.168.56.3:8888  │                 │             │
└─────────────────────┘                  └─────────────────────┘                  └─────────────┘
```

---

## 4. 使用说明

### 启动发送端 (ROS2)

```bash
ros2 run publisher_test udp_sender_node

# 或带参数启动
ros2 run publisher_test udp_sender_node --ros-args -p trigger_value:=5 -p auto_increment:=true
```

### 启动接收端 (Windows)

```bash
python windows_com_publisher.py
```

---

## 5. 注意事项

1. **网络配置：** 确保 ROS2 节点与 Windows 端在同一网络，IP 地址配置正确
2. **防火墙：** Windows 端需开放 UDP 8888 端口
3. **串口权限：** 确保 COM3 端口可用且未被其他程序占用
4. **Trigger 范围：** 有效值为 0-255，超出范围将被接收端拒绝
