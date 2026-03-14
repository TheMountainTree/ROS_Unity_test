#!/usr/bin/env python3
import socket

LISTEN_IP = "192.168.56.3"
LISTEN_PORT = 5005
BUFFER_SIZE = 4096


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, LISTEN_PORT))

    print(f"UDP receiver listening on {LISTEN_IP}:{LISTEN_PORT}")
    while True:
        data, addr = sock.recvfrom(BUFFER_SIZE)
        message = data.decode("utf-8", errors="replace")
        print(f"From {addr[0]}:{addr[1]} -> {message}")


if __name__ == "__main__":
    main()
