#!/usr/bin/env python3
"""
ROS2 Agent 工具模块
包含命令执行、响应解析等工具函数
"""

import re
import subprocess
import os
import signal
import base64
from typing import Dict, Optional, List

# 后台进程管理字典 {pid: process_info}
background_processes: Dict[int, dict] = {}


def execute_command(command: str, timeout: int = 30) -> dict:
    """
    执行Shell/ROS2命令
    
    Args:
        command: 要执行的命令字符串
        timeout: 超时时间（秒），默认30秒
    
    Returns:
        包含执行结果的字典:
        - success: 是否成功
        - output: 输出内容
        - return_code: 返回码
    """
    command = command.strip()
    if not command:
        return {"success": False, "output": "空命令"}
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            executable='/bin/bash'
        )
        
        output = result.stdout.strip() if result.stdout.strip() else result.stderr.strip()
        
        return {
            "success": result.returncode == 0,
            "output": output if output else "(无输出)",
            "return_code": result.returncode
        }
            
    except subprocess.TimeoutExpired:
        return {"success": False, "output": f"执行超时（{timeout}秒）"}
    except Exception as e:
        return {"success": False, "output": f"执行出错: {str(e)}"}


def parse_ai_response(response: str) -> dict:
    """
    解析AI响应，提取思考过程、命令和结束标记
    
    Args:
        response: AI的响应文本
    
    Returns:
        包含解析结果的字典:
        - think: 思考过程
        - command: 要执行的普通命令
        - background: 后台命令 {command, name}
        - stop_background: 停止后台进程 {pid}
        - list_background: 是否请求列出后台进程
        - read_images: 读取图片列表 [path1, path2, ...]
        - done: 是否任务完成
        - done_message: 完成消息
    """
    result = {
        "think": "",
        "command": "",
        "background": None,
        "stop_background": None,
        "list_background": False,
        "read_images": [],
        "done": False,
        "done_message": ""
    }
    
    # 提取思考过程
    think_match = re.search(r'<think]\s*(.*?)\s*</think', response, re.DOTALL | re.IGNORECASE)
    if think_match:
        result["think"] = think_match.group(1).strip()
    
    # 提取普通命令
    command_match = re.search(r'<command>\s*(.*?)\s*</command>', response, re.DOTALL | re.IGNORECASE)
    if command_match:
        result["command"] = command_match.group(1).strip()
    
    # 提取后台命令
    background_match = re.search(r'<background\s+name=["\']?(.*?)["\']?\s*>\s*(.*?)\s*</background>', response, re.DOTALL | re.IGNORECASE)
    if background_match:
        result["background"] = {
            "name": background_match.group(1).strip(),
            "command": background_match.group(2).strip()
        }
    
    # 提取停止后台进程命令
    stop_match = re.search(r'<stop_background\s+pid=["\']?(\d+)["\']?\s*>', response, re.IGNORECASE)
    if stop_match:
        result["stop_background"] = {
            "pid": int(stop_match.group(1))
        }
    
    # 检查是否请求列出后台进程
    if re.search(r'<list_background\s*>', response, re.IGNORECASE):
        result["list_background"] = True
    
    # 提取读取图片命令 <read_images>/path1.jpg,/path2.png</read_images>
    read_images_match = re.search(r'<read_images>\s*(.*?)\s*</read_images>', response, re.DOTALL | re.IGNORECASE)
    if read_images_match:
        paths_text = read_images_match.group(1).strip()
        # 支持逗号分隔或换行分隔
        paths = []
        for p in re.split(r'[,\n]+', paths_text):
            p = p.strip()
            if p:
                paths.append(p)
        result["read_images"] = paths
    
    # 检查是否完成
    done_match = re.search(r'<done>\s*(.*?)\s*</done>', response, re.DOTALL | re.IGNORECASE)
    if done_match:
        result["done"] = True
        result["done_message"] = done_match.group(1).strip()
    
    return result


def start_background_process(command: str, name: str = "") -> dict:
    """
    启动后台进程（适用于持续运行的命令，如ROS节点、launch文件等）
    
    Args:
        command: 要执行的命令字符串
        name: 进程名称（可选，用于标识）
    
    Returns:
        包含启动结果的字典:
        - success: 是否成功启动
        - pid: 进程ID
        - name: 进程名称
        - output: 状态信息
    """
    command = command.strip()
    if not command:
        return {"success": False, "output": "空命令"}
    
    try:
        # 使用 Popen 启动后台进程
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            executable='/bin/bash',
            # 设置为新进程组，避免终端信号影响
            start_new_session=True
        )
        
        pid = process.pid
        process_name = name if name else f"process_{pid}"
        
        # 记录进程信息
        background_processes[pid] = {
            "process": process,
            "command": command,
            "name": process_name,
            "status": "running"
        }
        
        return {
            "success": True,
            "pid": pid,
            "name": process_name,
            "command": command,
            "output": f"后台进程已启动 (PID: {pid}, 名称: {process_name})"
        }
        
    except Exception as e:
        return {"success": False, "output": f"启动后台进程失败: {str(e)}"}


def stop_background_process(pid: int) -> dict:
    """
    停止后台进程
    
    Args:
        pid: 进程ID
    
    Returns:
        包含停止结果的字典:
        - success: 是否成功停止
        - output: 状态信息
    """
    if pid not in background_processes:
        return {"success": False, "output": f"未找到PID为 {pid} 的后台进程"}
    
    try:
        process_info = background_processes[pid]
        process = process_info["process"]
        
        # 尝试终止整个进程组
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            # 如果进程组不存在，尝试直接终止进程
            process.terminate()
        
        # 等待进程结束
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # 如果还没结束，强制杀死
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                process.kill()
        
        process_info["status"] = "stopped"
        del background_processes[pid]
        
        return {
            "success": True,
            "output": f"后台进程已停止 (PID: {pid}, 名称: {process_info['name']})"
        }
        
    except Exception as e:
        return {"success": False, "output": f"停止后台进程失败: {str(e)}"}


def list_background_processes() -> dict:
    """
    列出所有后台进程
    
    Returns:
        包含进程列表的字典:
        - success: 是否成功
        - processes: 进程列表
        - output: 状态信息
    """
    if not background_processes:
        return {"success": True, "processes": [], "output": "当前没有后台进程"}
    
    processes = []
    for pid, info in background_processes.items():
        # 检查进程是否还在运行
        poll_result = info["process"].poll()
        is_running = poll_result is None
        
        processes.append({
            "pid": pid,
            "name": info["name"],
            "command": info["command"],
            "status": "running" if is_running else "stopped",
            "exit_code": poll_result
        })
    
    return {
        "success": True,
        "processes": processes,
        "output": f"共有 {len(processes)} 个后台进程"
    }


def load_image_as_base64(image_path: str) -> dict:
    """
    加载图片并转换为base64编码
    
    Args:
        image_path: 图片文件路径
    
    Returns:
        包含图片信息的字典:
        - success: 是否成功
        - base64: base64编码的图片数据
        - filename: 文件名
        - error: 错误信息（如果失败）
    """
    try:
        # 获取文件名（保持原始名称）
        filename = os.path.basename(image_path)
        
        # 读取图片文件
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # 转换为base64
        base64_data = base64.b64encode(image_data).decode('utf-8')
        
        return {
            "success": True,
            "base64": base64_data,
            "filename": filename
        }
        
    except FileNotFoundError:
        return {"success": False, "error": f"图片文件不存在: {image_path}"}
    except Exception as e:
        return {"success": False, "error": f"加载图片失败: {str(e)}"}


def load_images_as_message(image_paths: List[str], text: str = "") -> dict:
    """
    加载多张图片并转换为OpenAI API消息格式
    
    Args:
        image_paths: 图片路径列表
        text: 附加的文本描述
    
    Returns:
        包含消息内容的字典:
        - success: 是否成功
        - message: OpenAI API格式的消息内容
        - filenames: 成功加载的文件名列表
        - errors: 错误信息列表
    """
    content = []
    filenames = []
    errors = []
    
    # 添加文本部分
    if text:
        content.append({"type": "text", "text": text})
    
    # 加载每张图片
    for image_path in image_paths:
        result = load_image_as_base64(image_path)
        
        if result["success"]:
            # 获取图片类型
            ext = os.path.splitext(result["filename"])[1].lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }.get(ext, 'image/jpeg')
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{result['base64']}"
                }
            })
            filenames.append(result["filename"])
        else:
            errors.append(result["error"])
    
    if not filenames:
        return {
            "success": False,
            "message": None,
            "filenames": [],
            "errors": errors if errors else ["没有成功加载任何图片"]
        }
    
    return {
        "success": True,
        "message": {"role": "user", "content": content},
        "filenames": filenames,
        "errors": errors
    }


def write_file(file_path: str, content: str, mode: str = "write") -> dict:
    """
    写入文件
    
    Args:
        file_path: 文件路径
        content: 要写入的内容
        mode: 写入模式 - "write"（覆盖写入，默认）或 "append"（追加写入）
    
    Returns:
        包含写入结果的字典:
        - success: 是否成功
        - output: 状态信息
        - file_path: 文件绝对路径
        - bytes_written: 写入的字节数
    """
    try:
        # 获取绝对路径
        abs_path = os.path.abspath(file_path)
        
        # 确保父目录存在
        parent_dir = os.path.dirname(abs_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        # 根据模式选择写入方式
        write_mode = 'w' if mode == "write" else 'a'
        
        with open(abs_path, write_mode, encoding='utf-8') as f:
            bytes_written = f.write(content)
        
        action = "覆盖写入" if mode == "write" else "追加写入"
        return {
            "success": True,
            "output": f"文件{action}成功: {abs_path} ({bytes_written} 字符)",
            "file_path": abs_path,
            "bytes_written": bytes_written
        }
        
    except PermissionError:
        return {"success": False, "output": f"权限不足，无法写入文件: {file_path}"}
    except Exception as e:
        return {"success": False, "output": f"写入文件失败: {str(e)}"}


def read_file(file_path: str) -> dict:
    """
    读取文件内容
    
    Args:
        file_path: 文件路径
    
    Returns:
        包含读取结果的字典:
        - success: 是否成功
        - content: 文件内容
        - output: 状态信息
    """
    try:
        abs_path = os.path.abspath(file_path)
        
        if not os.path.exists(abs_path):
            return {"success": False, "output": f"文件不存在: {abs_path}"}
        
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "success": True,
            "content": content,
            "output": f"文件读取成功: {abs_path} ({len(content)} 字符)",
            "file_path": abs_path
        }
        
    except PermissionError:
        return {"success": False, "output": f"权限不足，无法读取文件: {file_path}"}
    except UnicodeDecodeError:
        return {"success": False, "output": f"文件编码不支持（非UTF-8文本文件）: {file_path}"}
    except Exception as e:
        return {"success": False, "output": f"读取文件失败: {str(e)}"}


def stream_ai_response(client, messages: list, model: str) -> str:
    """
    流式输出AI响应
    
    Args:
        client: OpenAI客户端实例
        messages: 消息历史列表
        model: 使用的模型名称
    
    Returns:
        AI的完整响应文本
    """
    ai_response = ""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                ai_response += content
        
        print()
        return ai_response
        
    except Exception as e:
        print(f"\n❌ API调用失败: {str(e)}")
        return ""


# 可用的工具列表
AVAILABLE_TOOLS = {
    "execute_command": {
        "function": execute_command,
        "description": "执行Shell或ROS2命令（同步执行，适用于快速完成的命令）",
        "parameters": {
            "command": "要执行的命令字符串",
            "timeout": "超时时间（秒），默认30"
        }
    },
    "start_background_process": {
        "function": start_background_process,
        "description": "启动后台进程（适用于持续运行的命令，如ROS节点、launch文件）",
        "parameters": {
            "command": "要执行的命令字符串",
            "name": "进程名称（可选，用于标识）"
        }
    },
    "stop_background_process": {
        "function": stop_background_process,
        "description": "停止后台进程",
        "parameters": {
            "pid": "要停止的进程ID"
        }
    },
    "list_background_processes": {
        "function": list_background_processes,
        "description": "列出所有后台进程",
        "parameters": {}
    },
    "load_image_as_base64": {
        "function": load_image_as_base64,
        "description": "加载图片并转换为base64编码",
        "parameters": {
            "image_path": "图片文件路径"
        }
    },
    "load_images_as_message": {
        "function": load_images_as_message,
        "description": "加载多张图片并转换为OpenAI API消息格式",
        "parameters": {
            "image_paths": "图片路径列表",
            "text": "附加的文本描述"
        }
    },
    "parse_ai_response": {
        "function": parse_ai_response,
        "description": "解析AI响应文本",
        "parameters": {
            "response": "AI的响应文本"
        }
    },
    "write_file": {
        "function": write_file,
        "description": "写入文件（支持覆盖写入和追加写入）",
        "parameters": {
            "file_path": "文件路径",
            "content": "要写入的内容",
            "mode": "写入模式：'write'（覆盖写入，默认）或 'append'（追加写入）"
        }
    },
    "read_file": {
        "function": read_file,
        "description": "读取文件内容",
        "parameters": {
            "file_path": "文件路径"
        }
    }
}
