#!/usr/bin/env python3
"""
EEG数据可视化脚本
绘制CSV文件中各个通道的数据图
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_eeg_data(csv_path, title=None):
    """
    绘制EEG数据各通道的波形图
    
    Args:
        csv_path: CSV文件路径
        title: 图表标题（可选）
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 获取通道名（排除timestamp和Trigger列）
    channels = [col for col in df.columns if col not in ['timestamp', 'Trigger']]
    
    # 创建子图
    fig, axes = plt.subplots(len(channels), 1, figsize=(14, 2.5 * len(channels)), sharex=True)
    
    # 如果只有一个通道，axes不是数组，需要转换
    if len(channels) == 1:
        axes = [axes]
    
    # 绘制每个通道的数据
    x = df.index.to_numpy()
    for i, channel in enumerate(channels):
        axes[i].plot(x, df[channel].values, linewidth=0.5, color='b')
        axes[i].set_ylabel(channel, fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, len(df))
    
    # 设置x轴标签
    axes[-1].set_xlabel('Sample Index', fontsize=10)
    
    # 设置标题
    if title is None:
        title = os.path.basename(csv_path)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig, axes


def plot_eeg_comparison(csv_paths, channel=None):
    """
    比较多个CSV文件的数据
    
    Args:
        csv_paths: CSV文件路径列表
        channel: 要比较的通道名（可选，默认比较所有通道）
    """
    dfs = [pd.read_csv(path) for path in csv_paths]
    labels = [os.path.basename(path) for path in csv_paths]
    
    # 获取通道名
    channels = [col for col in dfs[0].columns if col not in ['timestamp', 'Trigger']]
    
    if channel:
        channels = [channel]
    
    fig, axes = plt.subplots(len(channels), 1, figsize=(14, 2.5 * len(channels)), sharex=True)
    
    if len(channels) == 1:
        axes = [axes]
    
    colors = ['b', 'r', 'g', 'm', 'c']
    
    for i, ch in enumerate(channels):
        for j, df in enumerate(dfs):
            color = colors[j % len(colors)]
            x = df.index.to_numpy()
            axes[i].plot(x, df[ch].values, linewidth=0.5, color=color, 
                         label=labels[j], alpha=0.7)
        axes[i].set_ylabel(ch, fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Sample Index', fontsize=10)
    fig.suptitle('EEG Data Comparison', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig, axes


def main():
    # 数据目录
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 查找CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("未找到CSV文件")
        return
    
    print(f"找到的CSV文件: {csv_files}")
    
    # 绘制每个CSV文件的数据
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        print(f"正在绘制: {csv_file}")
        
        fig, _ = plot_eeg_data(csv_path)
        output_file = os.path.join(data_dir, f"{os.path.splitext(csv_file)[0]}_plot.png")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"已保存: {output_file}")
        plt.close(fig)
    
    # 如果有多个CSV文件，绘制对比图
    if len(csv_files) > 1:
        csv_paths = [os.path.join(data_dir, f) for f in csv_files]
        print("正在绘制对比图...")
        
        fig, _ = plot_eeg_comparison(csv_paths)
        output_file = os.path.join(data_dir, "comparison_plot.png")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"已保存: {output_file}")
        plt.close(fig)
    
    print("绘图完成！")


if __name__ == "__main__":
    main()