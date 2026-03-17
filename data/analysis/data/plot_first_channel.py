#!/usr/bin/env python3
"""
EEG数据可视化脚本 - 仅绘制第一通道
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_first_channel(csv_path):
    """
    绘制EEG数据第一通道的波形图
    
    Args:
        csv_path: CSV文件路径
    """
    # 读取CSV文件（只读取前1000行）
    df = pd.read_csv(csv_path, nrows=1000)
    
    # 获取通道名（排除timestamp和Trigger列）
    channels = [col for col in df.columns if col not in ['timestamp', 'Trigger']]
    
    if not channels:
        print(f"未找到数据通道: {csv_path}")
        return None, None
    
    # 获取第一通道
    first_channel = channels[0]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # 绘制数据
    x = df.index.to_numpy()
    ax.plot(x, df[first_channel].values, linewidth=0.5, color='b')
    ax.set_ylabel(first_channel, fontsize=10)
    ax.set_xlabel('Sample Index', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    title = os.path.basename(csv_path)
    fig.suptitle(f'{title} - {first_channel}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig, first_channel


def main():
    # 数据目录
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 查找CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("未找到CSV文件")
        return
    
    print(f"找到的CSV文件: {csv_files}")
    
    # 绘制每个CSV文件的第一通道
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        print(f"正在绘制: {csv_file}")
        
        fig, channel = plot_first_channel(csv_path)
        
        if fig is not None:
            output_file = os.path.join(data_dir, f"{os.path.splitext(csv_file)[0]}_first_channel.png")
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"已保存: {output_file} (通道: {channel})")
            plt.close(fig)
    
    print("绘图完成！")


if __name__ == "__main__":
    main()