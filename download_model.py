#!/usr/bin/env python3
"""
Pre-download InstanSeg model

Run this script to pre-download model,Avoid download failure when worker starts.
"""

import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("  InstanSeg model pre-download tool")
print("="*60)
print()

try:
    from instanseg import InstanSeg
    import torch
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    # Try to download model(According to InstanSeg official API)
    model_types_to_try = [
        "brightfield_nuclei",
        "fluorescence_nuclei", 
        "fluorescence_cells"
    ]
    image_reader = "openslide"  # for files
    
    for model_type in model_types_to_try:
        try:
            print(f"📥 Downloading model...")
            # Correct API: InstanSeg(model_name, image_reader=..., verbosity=...)
            instanseg = InstanSeg(model_type, image_reader=image_reader, verbosity=1)
            print(f"✅ {model_type} Model download successful")
            break
        except Exception as e:
            print(f"❌ {model_type} Model download failed: {str(e)[:150]}")
            continue
    
    print()
    print("="*60)
    print("✅ Model download complete")
    print("="*60)
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please ensure installed instanseg-torch: pip install instanseg-torch")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

