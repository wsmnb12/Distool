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
import urllib.request
import zipfile


addon_dir = os.path.dirname(__file__)
lib_dir = os.path.join(addon_dir, "lib")

# Ensure lib path is added to Python path
if lib_dir not in sys.path:
    sys.path.append(lib_dir)


def download_and_extract_lib():
    if os.path.exists(lib_dir):
        return  # already exists

    zip_url = "https://github.com/wsmnb12/Distool-/releases/download/libs/libs.zip" 
    tmp_zip = os.path.join(addon_dir, "lib.zip")

    try:
        print(f"[Distool] Downloading dependencies from {zip_url} ...")
        urllib.request.urlretrieve(zip_url, tmp_zip)
        with zipfile.ZipFile(tmp_zip, 'r') as zip_ref:
            zip_ref.extractall(addon_dir)
        os.remove(tmp_zip)
        print("[Distool] Dependencies installed successfully.")
    except Exception as e:
        print(f"[Distool] Failed to download/extract lib.zip: {e}")


download_and_extract_lib()


try:
    import numpy
    import scipy
    import cv2
except ImportError as e:
    print(f"[Distool] Error importing dependencies after install: {e}")


from . import distool_main

def register():
    distool_main.register()

def unregister():
    distool_main.unregister()
