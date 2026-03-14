import socket
import serial
import time

# Configuration parameters
UDP_IP = "192.168.56.3"
UDP_PORT = 8888
COM_PORT = "COM3"
BAUD_RATE = 115200

def main():
    # 1. Initialize UDP socket
    sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"[*] 正在监听 UDP 端口 {UDP_PORT}...")

    # 2. Initialize COM port
    try:
        ser = serial.Serial(
            port=COM_PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0
        )
        print(f"[*] 成功打开 COM 端口 {COM_PORT}，波特率 {BAUD_RATE}")
    except Exception as e:
        print(f"[!] 打开 COM 端口 {COM_PORT} 失败: {e}")
        return
    
    # 3. 监听循环并转发
    print("[*] 等待接受 Trigger 信号 ...")
    try:
        while True:
            # 接受UDP数据
            data, addr = sock.recvfrom(1024)

            if data:
                trigger_value = int.from_bytes(data, byteorder='little')

                if 0 <= trigger_value <= 255:
                    dcp_command = bytearray([0x01, 0xE1, 0x01, 0x00, trigger_value])

                    ser.write(dcp_command)
                    ser.flush()
                    print(f"[{time.time():.4f}] 已转发 Trigger: {trigger_value} (HEX: {trigger_value:02X})  来自 {addr}")
                else:
                    print(f"[!] 无效 Trigger 值: {trigger_value} (来自 {addr})")
    except KeyboardInterrupt:
        print("[*] 程序手动终止")
    finally:
        sock.close()
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print(f"[*] 已关闭 COM 端口 {COM_PORT}")
    
if __name__ == "__main__":
    main()
