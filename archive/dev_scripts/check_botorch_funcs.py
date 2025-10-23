#!/usr/bin/env python3
import bomegabench as bmb

# BoTorch中的函数（不包括约束函数）
botorch_funcs = [
    'Bukin', 'Cosine8', 'ThreeHumpCamel', 'AckleyMixed', 'Labs'
]

# 检查我们是否已经实现了这些函数
all_funcs = bmb.list_functions()
print('Current functions:', len(all_funcs))

print('\nChecking BoTorch functions:')
missing_funcs = []
for func in botorch_funcs:
    found = [f for f in all_funcs if func.lower() in f.lower()]
    if found:
        print(f'{func}: FOUND - {found}')
    else:
        print(f'{func}: NOT FOUND')
        missing_funcs.append(func)

print(f'\nMissing functions: {missing_funcs}')
print(f'Need to implement: {len(missing_funcs)} functions') 