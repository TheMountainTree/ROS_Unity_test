import socket
import struct
import time
import csv
from pathlib import Path

class WindowsEegTcpClient:
    def __init__(self, server_ip="127.0.0.1", server_port=8712):
        # 核心参数配置
        self.server_ip = server_ip
        self.server_port = server_port
        self.eeg_sample_rate_hz = 1000.0  # 假设采样率为 1000Hz
        self.dt = 1.0 / self.eeg_sample_rate_hz
        
        # 通道设置 (8个脑电通道 + 1个Trigger)
        self.channel_names = ["POz", "PO3", "PO4", "PO5", "PO6", "Oz", "O1", "O2"]
        self.n_eeg_channels = len(self.channel_names)
        self.frame_floats = self.n_eeg_channels + 1
        self.frame_bytes = self.frame_floats * 4     # 36 字节一帧
        self.unpack_fmt = f"<{self.frame_floats}f"   # 强制使用小端序解包
        
        # 状态变量
        self.buffer = bytearray()
        self.frame_synced = False
        self.total_frames_received = 0
        
        # 初始化 CSV 文件
        self.csv_path = Path("eeg_local_test_data.csv")
        file_exists = self.csv_path.exists()
        self.csv_file = open(self.csv_path, mode="a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        if not file_exists:
            self.csv_writer.writerow(["timestamp"] + self.channel_names + ["Trigger"])
            self.csv_file.flush()

        # 初始化 TCP 客户端
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2.0)
        
    def connect_and_listen(self):
        print(f"[*] 正在连接到脑电 TCP 服务端 {self.server_ip}:{self.server_port}...")
        try:
            self.sock.connect((self.server_ip, self.server_port))
            # 禁用 Nagle 算法，确保底层网络数据包极速推送到应用层
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # 将超时时间改短，模拟非阻塞轮询
            self.sock.settimeout(0.01)
            print("[+] 连接成功！开始抓取数据...\n")
        except Exception as e:
            print(f"[-] 连接失败: {e}")
            return

        try:
            while True:
                self._poll_tcp()
                # 适度休眠，避免单核 CPU 被 while 循环 100% 占满
                time.sleep(0.005) 
        except KeyboardInterrupt:
            print("\n[*] 检测到 Ctrl+C，正在安全终止并保存数据...")
        finally:
            self.shutdown()

    def _poll_tcp(self):
        # 1. 读取数据到缓存
        try:
            while True:
                chunk = self.sock.recv(4096)
                if not chunk:
                    print("[-] 服务端主动断开了连接。")
                    return
                self.buffer.extend(chunk)
        except socket.timeout:
            pass # 缓冲区暂无新数据，跳出继续处理已有 buffer
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"[-] 接收数据异常: {e}")
            return

        # 2. 帧头双重验证与同步猎捕 (解决错位与偶尔为0.0造成的 False Sync)
        if not self.frame_synced:
            sync_pos = self.buffer.find(b"\x00\x00\x00\x00")
            if sync_pos < 0:
                if len(self.buffer) > 3:
                    del self.buffer[:-3] # 保留末尾3字节防截断
                return
            
            expected_next_trigger = sync_pos + self.frame_bytes
            
            # 等待接收足够的数据以验证下一个周期
            if len(self.buffer) < expected_next_trigger + 4:
                return 

            # 验证下一个周期的对应位置是否也是 Trigger (0.0)
            if self.buffer[expected_next_trigger : expected_next_trigger+4] == b"\x00\x00\x00\x00":
                frame_start = sync_pos + 4
                del self.buffer[:frame_start]
                self.frame_synced = True
                print("[!] 成功捕获帧头，双重验证通过！数据流已对齐。")
            else:
                # 这是一个“伪 Trigger”（某个脑电通道刚好等于0.0），丢弃当前这个0并继续寻找
                del self.buffer[:sync_pos + 1]
                return

        # 3. 提取完整帧并进行时间戳插值
        num_complete_frames = len(self.buffer) // self.frame_bytes
        if num_complete_frames == 0:
            return

        now = time.time()
        # 核心修复：无论收到多少帧（只要>1），一律进行线性时间戳回溯
        use_ts_interp = (num_complete_frames > 1)

        for i in range(num_complete_frames):
            frame_raw = self.buffer[:self.frame_bytes]
            del self.buffer[:self.frame_bytes]
            
            vals = struct.unpack(self.unpack_fmt, frame_raw)
            
            if use_ts_interp:
                # 假设当前时间 now 对应这一批次中【最后一帧】到达的时间
                # 那么第 i 帧的时间 = now - (总帧数 - 1 - i) * dt
                ts = now - ((num_complete_frames - 1 - i) * self.dt)
            else:
                ts = now
                
            # 将 Trigger 从浮点数强转为整数，避免 255.0 这种形式
            trigger_val = int(round(vals[-1]))
            row = [f"{ts:.6f}"] + [f"{v:.4f}" for v in vals[:-1]] + [str(trigger_val)]
            self.csv_writer.writerow(row)
            
        self.csv_file.flush()
        self.total_frames_received += num_complete_frames
        
        if self.total_frames_received % 1000 == 0:
            print(f"已接收 {self.total_frames_received} 帧数据... (Buffer 余量: {len(self.buffer)} bytes)")

    def shutdown(self):
        self.sock.close()
        if self.csv_file:
            self.csv_file.close()
        print("[*] 资源已释放，程序安全退出。")

if __name__ == "__main__":
    # 提醒：如果是同一台电脑，请保持 127.0.0.1
    # 如果是局域网内其他电脑，请将 127.0.0.1 替换为采集软件所在电脑的 IP
    client = WindowsEegTcpClient(server_ip="127.0.0.1", server_port=8712)
    client.connect_and_listen()