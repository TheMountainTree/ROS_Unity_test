#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI ROS2 Agent - 交互式ROS2控制Agent
支持多轮对话、上下文记忆、自动迭代执行直到任务完成
"""

import os
import httpx
from openai import OpenAI
from tools import (
    execute_command, 
    parse_ai_response, 
    stream_ai_response,
    start_background_process,
    stop_background_process,
    list_background_processes,
    load_images_as_message
)

# ============== 配置区域 ==============
BASE_URL = "https://cloud.infini-ai.com/maas/coding/v1"
API_KEY = "sk-cp-7udwseww3rks2yfx"
MODEL = "kimi-k2.5"

# Agent自动迭代的最大轮次
MAX_AGENT_TURNS = 20

# 文件路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_FILE = os.path.join(CURRENT_DIR, "SystemPrompt.md")
KNOWLEDGE_FILE = os.path.join(CURRENT_DIR, "Knowledge.md")
# =====================================


def load_markdown_file(filepath: str) -> str:
    """
    加载Markdown文件内容
    
    Args:
        filepath: 文件路径
    
    Returns:
        文件内容字符串
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"⚠️ 文件不存在: {filepath}")
        return ""
    except Exception as e:
        print(f"⚠️ 加载文件失败 {filepath}: {str(e)}")
        return ""


def build_system_prompt() -> str:
    """
    构建完整的系统提示词（包含系统提示和背景知识）
    
    Returns:
        完整的系统提示词
    """
    # 加载系统提示词
    system_prompt = load_markdown_file(SYSTEM_PROMPT_FILE)
    
    # 加载背景知识
    knowledge = load_markdown_file(KNOWLEDGE_FILE)
    
    # 组合系统提示词
    if knowledge:
        full_prompt = f"{system_prompt}\n\n---\n\n# 背景知识\n\n{knowledge}"
    else:
        full_prompt = system_prompt
    
    return full_prompt


def run_agent_loop(client: OpenAI, messages: list, user_input: str):
    """
    运行Agent循环：执行命令 -> 返回结果 -> 继续执行，直到任务完成
    """
    # 添加用户消息
    messages.append({"role": "user", "content": user_input})
    
    turn = 0
    while turn < MAX_AGENT_TURNS:
        turn += 1
        
        # 流式输出AI响应
        print(f"\n\033[1;32mAI\033[0m> ", end="", flush=True)
        ai_response = stream_ai_response(client, messages, MODEL)
        
        if not ai_response:
            messages.pop()
            return
        
        # 添加AI响应到历史
        messages.append({"role": "assistant", "content": ai_response})
        
        # 解析AI响应
        parsed = parse_ai_response(ai_response)
        
        # 显示思考过程（如果有）
        if parsed["think"]:
            print(f"\033[90m💭 思考: {parsed['think']}\033[0m")
        
        # 检查是否完成
        if parsed["done"]:
            print(f"\n\033[1;34m✨ 任务完成: {parsed['done_message']}\033[0m")
            return
        
        # 收集执行结果
        feedback_parts = []
        has_action = False
        
        # 执行普通命令
        if parsed["command"]:
            has_action = True
            print(f"\n\033[1;33m⚡ 执行命令\033[0m: {parsed['command']}")
            result = execute_command(parsed["command"])
            
            if result["success"]:
                print(f"\033[1;32m✅ 成功\033[0m")
            else:
                print(f"\033[1;31m❌ 失败\033[0m")
            
            if result["output"]:
                print(f"\033[90m{result['output']}\033[0m")
            
            feedback_parts.append(f"[系统] 普通命令执行结果:\n成功: {result['success']}\n输出: {result['output']}")
        
        # 执行后台命令
        if parsed["background"]:
            has_action = True
            bg = parsed["background"]
            print(f"\n\033[1;35m🚀 启动后台进程\033[0m [{bg['name']}]: {bg['command']}")
            result = start_background_process(bg["command"], bg["name"])
            
            if result["success"]:
                print(f"\033[1;32m✅ 后台进程已启动 (PID: {result['pid']})\033[0m")
            else:
                print(f"\033[1;31m❌ 启动失败\033[0m")
            
            feedback_parts.append(f"[系统] 后台进程启动结果:\n成功: {result['success']}\nPID: {result.get('pid', 'N/A')}\n信息: {result['output']}")
        
        # 停止后台进程
        if parsed["stop_background"]:
            has_action = True
            pid = parsed["stop_background"]["pid"]
            print(f"\n\033[1;31m🛑 停止后台进程\033[0m (PID: {pid})")
            result = stop_background_process(pid)
            
            if result["success"]:
                print(f"\033[1;32m✅ 已停止\033[0m")
            else:
                print(f"\033[1;31m❌ 停止失败\033[0m")
            
            feedback_parts.append(f"[系统] 停止后台进程结果:\n成功: {result['success']}\n信息: {result['output']}")
        
        # 列出后台进程
        if parsed["list_background"]:
            has_action = True
            print(f"\n\033[1;36m📋 后台进程列表\033[0m")
            result = list_background_processes()
            
            if result["processes"]:
                for p in result["processes"]:
                    status_icon = "🟢" if p["status"] == "running" else "🔴"
                    print(f"  {status_icon} PID: {p['pid']} | 名称: {p['name']} | 命令: {p['command']}")
            else:
                print("  (无后台进程)")
            
            feedback_parts.append(f"[系统] 后台进程列表:\n{result['output']}\n进程数: {len(result['processes'])}")
        
        # Agent主动读取图片
        if parsed["read_images"]:
            has_action = True
            image_paths = parsed["read_images"]
            print(f"\n\033[1;34m📷 读取图片中...\033[0m")
            
            # 加载图片
            result = load_images_as_message(image_paths, "这是请求读取的图片，请分析图片内容。")
            
            if result["success"]:
                print(f"\033[1;32m✅ 已读取图片: {', '.join(result['filenames'])}\033[0m")
                
                # 创建一个临时消息列表用于图片分析
                temp_messages = list(messages)  # 复制当前消息历史
                temp_messages.append(result["message"])
                
                # 流式输出AI响应
                print(f"\n\033[1;32mAI\033[0m> ", end="", flush=True)
                image_response = stream_ai_response(client, temp_messages, MODEL)
                
                if image_response:
                    # 将图片和AI的分析添加到消息历史
                    messages.append(result["message"])
                    messages.append({"role": "assistant", "content": image_response})
                
                feedback_parts.append(f"[系统] 图片读取结果:\n成功: True\n图片: {', '.join(result['filenames'])}\nAI已分析图片内容")
                
                # 显示错误（如果有）
                if result["errors"]:
                    for err in result["errors"]:
                        print(f"\033[1;33m⚠️ {err}\033[0m")
                        feedback_parts.append(f"警告: {err}")
            else:
                print(f"\033[1;31m❌ 读取图片失败:\033[0m")
                for err in result["errors"]:
                    print(f"  {err}")
                feedback_parts.append(f"[系统] 图片读取失败:\n" + "\n".join(result["errors"]))
        
        # 如果有执行动作，将结果返回给AI
        if has_action:
            feedback = "\n\n".join(feedback_parts)
            messages.append({"role": "user", "content": feedback})
        else:
            # AI没有输出命令也没有标记完成，结束循环
            print("\n\033[1;33m⚠️ AI未输出可执行的命令\033[0m")
            return
    
    print(f"\n\033[1;33m⚠️ 达到最大迭代轮次 ({MAX_AGENT_TURNS})，请继续提问让AI完成任务\033[0m")


def parse_image_command(user_input: str):
    """
    解析图片输入命令
    
    格式: /image 图片路径1,图片路径2 你的问题
    
    Returns:
        (image_paths, text) 或 None
    """
    if not user_input.startswith('/image '):
        return None
    
    # 移除 /image 前缀
    rest = user_input[7:].strip()
    
    # 查找第一个空格，分割路径和文本
    parts = rest.split(None, 1)
    
    if not parts:
        return None
    
    # 解析图片路径（逗号分隔）
    image_paths = [p.strip() for p in parts[0].split(',') if p.strip()]
    
    # 获取文本部分
    text = parts[1] if len(parts) > 1 else "请描述这些图片内容"
    
    return image_paths, text


def run_interactive_session(client: OpenAI):
    """运行交互式对话会话"""
    # 构建系统提示词
    system_prompt = build_system_prompt()
    
    print("\n" + "=" * 60)
    print("🤖 AI ROS2 Agent - 交互模式")
    print("=" * 60)
    print("💡 输入你的问题，AI会自动执行命令直到任务完成")
    print("📌 输入 'exit' 或 'quit' 退出")
    print("📌 输入 'clear' 清除对话历史")
    print("📌 输入 '/image 图片路径1,图片路径2 问题' 发送图片给AI")
    print("=" * 60 + "\n")
    
    # 初始化消息历史（持久化上下文）
    messages = [{"role": "system", "content": system_prompt}]
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n\033[1;36m你\033[0m> ").strip()
            
            if not user_input:
                continue
            
            # 检查退出命令
            if user_input.lower() in ['exit', 'quit', 'q', '退出']:
                print("\n👋 再见！")
                break
            
            # 检查清除历史命令
            if user_input.lower() in ['clear', '清除']:
                messages = [{"role": "system", "content": system_prompt}]
                print("\n🗑️ 对话历史已清除")
                continue
            
            # 检查图片输入命令
            image_cmd = parse_image_command(user_input)
            if image_cmd:
                image_paths, text = image_cmd
                print(f"\n\033[1;34m📷 加载图片中...\033[0m")
                
                # 加载图片
                result = load_images_as_message(image_paths, text)
                
                if result["success"]:
                    print(f"\033[1;32m✅ 已加载图片: {', '.join(result['filenames'])}\033[0m")
                    
                    # 添加图片消息到历史
                    messages.append(result["message"])
                    
                    # 流式输出AI响应
                    print(f"\n\033[1;32mAI\033[0m> ", end="", flush=True)
                    ai_response = stream_ai_response(client, messages, MODEL)
                    
                    if ai_response:
                        messages.append({"role": "assistant", "content": ai_response})
                    
                    # 显示错误（如果有）
                    if result["errors"]:
                        for err in result["errors"]:
                            print(f"\033[1;33m⚠️ {err}\033[0m")
                else:
                    print(f"\033[1;31m❌ 加载图片失败:\033[0m")
                    for err in result["errors"]:
                        print(f"  {err}")
                continue
            
            # 运行Agent循环
            run_agent_loop(client, messages, user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")


def main():
    """主函数"""

    http_client = httpx.Client(
    proxy="socks5://127.0.0.1:7897"
    )
    
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        http_client=http_client
    )
    
    run_interactive_session(client)


if __name__ == "__main__":
    main()