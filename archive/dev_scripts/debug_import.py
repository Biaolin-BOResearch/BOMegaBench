#!/usr/bin/env python3
"""Debug import priority."""

print("Testing direct COCO import...")
try:
    from bomegabench.functions.bbob_coco import BBOBCOCOSuite
    print(f"✓ COCO suite imported: {len(BBOBCOCOSuite.functions)} functions")
    
    sphere_coco = BBOBCOCOSuite.get_function("sphere")
    print(f"COCO sphere type: {type(sphere_coco)}")
    print(f"COCO sphere suite: {sphere_coco.metadata['suite']}")
    print()
except Exception as e:
    print(f"✗ COCO import failed: {e}")
    import traceback
    traceback.print_exc()

print("Testing functions/__init__.py import...")
try:
    from bomegabench.functions import BBOBSuite
    print(f"✓ BBOBSuite imported: {len(BBOBSuite.functions)} functions")
    
    sphere_imported = BBOBSuite.get_function("sphere")
    print(f"Imported sphere type: {type(sphere_imported)}")
    print(f"Imported sphere suite: {sphere_imported.metadata['suite']}")
    print()
except Exception as e:
    print(f"✗ BBOBSuite import failed: {e}")
    import traceback
    traceback.print_exc()

print("Testing main bomegabench import...")
try:
    import bomegabench as bmb
    sphere_main = bmb.get_function("sphere")
    print(f"Main sphere type: {type(sphere_main)}")
    print(f"Main sphere suite: {sphere_main.metadata['suite']}")
except Exception as e:
    print(f"✗ Main import failed: {e}")
    import traceback
    traceback.print_exc() 