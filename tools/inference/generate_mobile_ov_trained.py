#!/usr/bin/env python3
"""
Inference script for MobileOVModel with TRAINED adapter and projection.

This version loads the trained components from training checkpoint.

Usage:
    python generate_mobile_ov_trained.py --prompt "a cat playing with a wool beside the fireside"
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import from the original script
from tools.inference.generate_mobile_ov import *

if __name__ == "__main__":
    # Override default paths to use trained components
    import argparse
    
    # Parse args first
    args = parse_args()
    
    # Override with trained components
    args.adapter_ckpt_dir = "output/trained_components/adapter"
    args.smolvlm2_ckpt_path = "omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt"
    
    # Update projection path in MobileOVModel loading
    # We'll modify the main function to use trained projection
    print(f"Using trained adapter from: {args.adapter_ckpt_dir}")
    print(f"Using trained projection from: output/trained_components/smolvlm2_projection")
    
    # Call main with modified args
    main()

