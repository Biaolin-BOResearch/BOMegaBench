#!/usr/bin/env python3
import bomegabench as bmb

# BoTorch中的所有SyntheticTestFunction类
botorch_all_funcs = [
    'Ackley', 'Beale', 'Branin', 'Bukin', 'Cosine8', 'DropWave', 
    'DixonPrice', 'EggHolder', 'Griewank', 'Hartmann', 'HolderTable',
    'Levy', 'Michalewicz', 'Powell', 'Rastrigin', 'Rosenbrock', 
    'Shekel', 'SixHumpCamel', 'StyblinskiTang', 'ThreeHumpCamel', 
    'AckleyMixed', 'Labs'
]

# 检查我们是否已经实现了这些函数
all_funcs = bmb.list_functions()
print(f'Current total functions: {len(all_funcs)}')

print('\nChecking all BoTorch functions:')
missing_funcs = []
found_funcs = []

for func in botorch_all_funcs:
    # 检查多种可能的名称变体
    found = []
    for f in all_funcs:
        if (func.lower() in f.lower() or 
            func.lower().replace('_', '') in f.lower().replace('_', '') or
            f.lower() in func.lower()):
            found.append(f)
    
    if found:
        print(f'{func}: FOUND - {found[:3]}{"..." if len(found) > 3 else ""}')
        found_funcs.append(func)
    else:
        print(f'{func}: NOT FOUND')
        missing_funcs.append(func)

print(f'\nSummary:')
print(f'Found: {len(found_funcs)} functions')
print(f'Missing: {len(missing_funcs)} functions')
print(f'Missing functions: {missing_funcs}')

# 检查各个套件的函数数量
classical = bmb.get_functions_by_property('suite', 'Classical')
bbob = bmb.get_functions_by_property('suite', 'BBOB Raw')
botorch = bmb.get_functions_by_property('suite', 'BoTorch')

print(f'\nSuite breakdown:')
print(f'Classical: {len(classical)}')
print(f'BBOB Raw: {len(bbob)}')
print(f'BoTorch: {len(botorch)}')
print(f'Total: {len(classical) + len(bbob) + len(botorch)}') 