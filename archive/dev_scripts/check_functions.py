#!/usr/bin/env python3

import bomegabench as bmb

print("All functions:", bmb.list_functions())
print()

print("Multimodal functions:", list(bmb.get_multimodal_functions().keys()))
print()

print("Unimodal functions:", list(bmb.get_unimodal_functions().keys()))
print()

# Check separable functions
try:
    separable = bmb.get_functions_by_property("properties", "separable")
    print("Separable functions:", list(separable.keys()))
except Exception as e:
    print("Error getting separable functions:", e)
print()

# Check a specific function's properties
sphere = bmb.get_function("sphere")
print("Sphere function properties:", sphere.metadata.get("properties", []))
print()

ellipsoid = bmb.get_function("ellipsoid_separable")
print("Ellipsoid separable properties:", ellipsoid.metadata.get("properties", []))
print() 