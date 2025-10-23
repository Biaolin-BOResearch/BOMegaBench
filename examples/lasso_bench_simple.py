#!/usr/bin/env python3
"""
简化的LassoBench集成示例

此示例展示如何使用BOMegaBench中集成的LassoBench函数。
注意：需要安装LassoBench及其依赖项。

安装命令：
pip install git+https://github.com/ksehic/LassoBench.git
"""

import numpy as np
import torch
import bomegabench as bmb

def main():
    print("LassoBench简化集成示例")
    print("=" * 40)
    
    # 检查可用的suites
    suites = bmb.list_suites()
    print(f"可用的suites: {suites}")
    
    lasso_suites = [s for s in suites if 'lasso' in s]
    if not lasso_suites:
        print("\n❌ 没有找到LassoBench suites")
        print("可能的原因：")
        print("1. LassoBench未安装")
        print("2. LassoBench依赖项缺失（如celer, sparse-ho等）")
        print("\n安装方法：")
        print("pip install git+https://github.com/ksehic/LassoBench.git")
        return
    
    print(f"\n✓ 找到LassoBench suites: {lasso_suites}")
    
    # 测试synthetic functions
    if 'lasso_synthetic' in lasso_suites:
        print("\n" + "="*50)
        print("测试LassoBench Synthetic Functions")
        print("="*50)
        
        try:
            functions = bmb.list_functions('lasso_synthetic')
            print(f"可用的synthetic functions: {functions}")
            
            # 测试最简单的function
            func_name = 'synt_simple_noiseless'
            if func_name in functions:
                func = bmb.get_function(func_name, 'lasso_synthetic')
                print(f"\n测试函数: {func.metadata['name']}")
                print(f"维度: {func.dim}")
                print(f"活跃维度: {func.metadata['active_dimensions']}")
                print(f"属性: {func.metadata['properties']}")
                
                # 生成随机测试点
                X = torch.rand(1, func.dim) * 2 - 1  # 缩放到[-1, 1]
                result = func(X)
                print(f"随机评估结果: {result.item():.6f}")
                
                # 获取测试指标
                if hasattr(func, 'get_test_metrics'):
                    metrics = func.get_test_metrics(X)
                    print(f"测试MSE: {metrics['mspe']:.6f}")
                    print(f"F-score: {metrics['fscore']:.6f}")
                    
        except Exception as e:
            print(f"测试synthetic functions时出错: {e}")
    
    # 测试real-world functions  
    if 'lasso_real' in lasso_suites:
        print("\n" + "="*50)
        print("测试LassoBench Real-world Functions")
        print("="*50)
        
        try:
            functions = bmb.list_functions('lasso_real')
            print(f"可用的real-world functions: {functions}")
            
            # 测试最小的dataset (diabetes)
            func_name = 'diabetes'
            if func_name in functions:
                func = bmb.get_function(func_name, 'lasso_real')
                print(f"\n测试函数: {func.metadata['name']}")
                print(f"维度: {func.dim}")
                print(f"数据集: {func.metadata['dataset']}")
                print(f"属性: {func.metadata['properties']}")
                
                # 生成随机测试点
                X = torch.rand(1, func.dim) * 2 - 1
                result = func(X)
                print(f"随机评估结果: {result.item():.6f}")
                
                # 获取测试指标
                if hasattr(func, 'get_test_metrics'):
                    metrics = func.get_test_metrics(X)
                    print(f"测试MSE: {metrics['mspe']:.6f}")
                    
        except Exception as e:
            print(f"测试real-world functions时出错: {e}")
    
    print("\n" + "="*40)
    print("示例完成！")


if __name__ == "__main__":
    main() 