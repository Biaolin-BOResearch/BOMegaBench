#!/usr/bin/env python3
import bomegabench as bmb

# Get all suites
classical = bmb.get_functions_by_property('suite', 'Classical')
bbob = bmb.get_functions_by_property('suite', 'BBOB Raw')
botorch = bmb.get_functions_by_property('suite', 'BoTorch')

print(f"Classical: {len(classical)}")
print(f"BBOB Raw: {len(bbob)}")
print(f"BoTorch: {len(botorch)}")
print(f"Total: {len(classical) + len(bbob) + len(botorch)}")

# Test Hartmann functions
try:
    h3d = bmb.get_function("hartmann_3d")
    h6d = bmb.get_function("hartmann_6d")
    print("Hartmann functions: OK")
except:
    print("Hartmann functions: ERROR")

# Count unique classical types
unique = set()
for name in classical.keys():
    base = name.replace('_3d', '').replace('_6d', '')
    unique.add(base)

print(f"Unique Classical types: {len(unique)}/42") 