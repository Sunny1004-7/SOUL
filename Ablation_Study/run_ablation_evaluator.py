# -*- coding: utf-8 -*-
"""
运行消融实验评估器的脚本
自动评估4种消融方法的教师回复质量
"""
import os
import sys
import argparse
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from ablation_evaluator import main as evaluate_main


def run_ablation_evaluation(data_dir: str = None, output_file: str = None, methods: list = None):
    """运行消融实验评估"""
    
    # 设置默认参数
    if data_dir is None:
        data_dir = os.path.dirname(__file__)
    
    if output_file is None:
        output_file = "ablation_evaluation_results.xlsx"
    
    if methods is None:
        methods = ["1", "2", "3", "4"]  # 默认评估所有4种消融方法
    
    # 构建命令行参数
    sys.argv = [
        "run_ablation_evaluator.py",
        "--data_dir", data_dir,
        "--output_file", output_file,
        "--methods"
    ] + methods
    
    print("="*60)
    print("🚀 开始运行消融实验评估")
    print("="*60)
    print(f"数据目录: {data_dir}")
    print(f"输出文件: {output_file}")
    print(f"评估方法: {', '.join(methods)}")
    print("="*60)
    
    try:
        # 调用主评估函数
        evaluate_main()
        
        print("\n" + "="*60)
        print("✅ 消融实验评估完成！")
        print(f"📊 结果已保存到: {output_file}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 评估过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行消融实验教师回复质量评估")
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录路径（默认为当前目录）")
    parser.add_argument("--output_file", type=str, default=None, help="输出文件路径（默认为ablation_evaluation_results.xlsx）")
    parser.add_argument("--methods", nargs="+", default=None, help="要评估的消融方法编号（默认为1 2 3 4）")
    parser.add_argument("--quick", action="store_true", help="快速模式，只评估方法1和2")
    
    args = parser.parse_args()
    
    # 快速模式
    if args.quick:
        methods = ["1", "2"]
    else:
        methods = args.methods
    
    run_ablation_evaluation(
        data_dir=args.data_dir,
        output_file=args.output_file,
        methods=methods
    )
