bl_info = {
    "name": "Distool: Displacement & Normal Generator",
    "author": "wsmnb12",
    "version": (1, 0, 0),
    "blender": (4, 4, 0),
    "location": "Shader Editor > Sidebar > Distool",
    "description": "Generate normal/displacement maps from textures with no setup needed.",
    "category": "Node",
}

import sys
import os

# Path to internal lib folder bundled with addon
addon_dir = os.path.dirname(__file__)
lib_dir = os.path.join(addon_dir, "lib")

if lib_dir not in sys.path:
    sys.path.append(lib_dir)

try:
    import numpy
    import scipy
    import cv2
except ImportError:
    print("[Distool] Required dependencies not found or failed to load.")

from . import distool_main

def register():
    distool_main.register()

def unregister():
    distool_main.unregister()
