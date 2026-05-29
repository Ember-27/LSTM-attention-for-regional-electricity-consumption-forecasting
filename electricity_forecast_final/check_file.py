"""
文件信息查看器
用于显示项目中所有的 .pkl 和 .npy 文件的信息
包括文件大小、形状（对于.npy文件）、数据类型等
"""

import os
import numpy as np
import pickle
import pandas as pd
from datetime import datetime


def format_file_size(size_bytes):
    """将文件大小从字节转换为可读格式"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f}{size_names[i]}"


def check_npy_file(file_path):
    """检查 .npy 文件并显示详细信息"""
    print(f"\n{'=' * 60}")
    print(f"文件: {os.path.basename(file_path)}")
    print(f"{'=' * 60}")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"  状态: 文件不存在")
        return

    # 显示文件基本信息
    file_size = os.path.getsize(file_path)
    mod_time = os.path.getmtime(file_path)
    mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

    print(f"  文件大小: {format_file_size(file_size)}")
    print(f"  修改时间: {mod_time_str}")

    # 尝试加载 .npy 文件
    try:
        data = np.load(file_path)
        print(f"  数据类型: {type(data)}")

        if isinstance(data, np.ndarray):
            print(f"  数组形状: {data.shape}")
            print(f"  数组数据类型: {data.dtype}")
            print(f"  数组维数: {data.ndim}")
            print(f"  元素总数: {data.size}")

            # 显示数据统计信息
            if data.size > 0:
                print(f"  最小值: {data.min()}")
                print(f"  最大值: {data.max()}")
                print(f"  平均值: {data.mean():.6f}")

                # 显示前几个元素
                print(f"  前5个元素: ", end="")
                if data.ndim == 1:
                    print(data[:5])
                elif data.ndim == 2:
                    print(data[:2, :5])  # 显示前2行，前5列
                else:
                    print(data.flat[:5])  # 显示前5个扁平化元素
        else:
            print(f"  数据内容: {data}")

    except Exception as e:
        print(f"  加载错误: {str(e)}")


def check_pkl_file(file_path):
    """检查 .pkl 文件并显示详细信息"""
    print(f"\n{'=' * 60}")
    print(f"文件: {os.path.basename(file_path)}")
    print(f"{'=' * 60}")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"  状态: 文件不存在")
        return

    # 显示文件基本信息
    file_size = os.path.getsize(file_path)
    mod_time = os.path.getmtime(file_path)
    mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

    print(f"  文件大小: {format_file_size(file_size)}")
    print(f"  修改时间: {mod_time_str}")

    # 尝试加载 .pkl 文件
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"  数据类型: {type(data)}")

        # 根据数据类型显示不同信息
        if isinstance(data, pd.DataFrame):
            print(f"  DataFrame 形状: {data.shape}")
            print(f"  列名: {list(data.columns)}")
            print(f"  前5行数据:")
            print(data.head())
        elif isinstance(data, dict):
            print(f"  字典键: {list(data.keys())}")
            print(f"  字典大小: {len(data)}")
        elif isinstance(data, list):
            print(f"  列表长度: {len(data)}")
            if len(data) > 0:
                print(f"  第一个元素类型: {type(data[0])}")
        elif isinstance(data, np.ndarray):
            print(f"  数组形状: {data.shape}")
            print(f"  数组数据类型: {data.dtype}")
        else:
            print(f"  数据内容: {data}")

    except Exception as e:
        print(f"  加载错误: {str(e)}")


def main():
    """主函数：检查当前目录下的所有 .npy 和 .pkl 文件"""
    print("=" * 60)
    print("文件信息查看器")
    print("=" * 60)

    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前目录: {current_dir}")

    # 查找所有 .npy 和 .pkl 文件
    npy_files = []
    pkl_files = []

    for file in os.listdir(current_dir):
        if file.endswith('.npy'):
            npy_files.append(file)
        elif file.endswith('.pkl'):
            pkl_files.append(file)

    # 显示 .npy 文件信息
    if npy_files:
        print(f"\n找到 {len(npy_files)} 个 .npy 文件:")
        for file in sorted(npy_files):
            file_path = os.path.join(current_dir, file)
            check_npy_file(file_path)
    else:
        print("\n未找到 .npy 文件")

    # 显示 .pkl 文件信息
    if pkl_files:
        print(f"\n找到 {len(pkl_files)} 个 .pkl 文件:")
        for file in sorted(pkl_files):
            file_path = os.path.join(current_dir, file)
            check_pkl_file(file_path)
    else:
        print("\n未找到 .pkl 文件")

    print("\n" + "=" * 60)
    print("检查完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
