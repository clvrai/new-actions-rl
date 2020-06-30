from ..constants import asset_dir
from PIL import Image
import os.path as osp
import os

img_library = {}

def get_loaded_images(img_dict):
    for filename in os.listdir(asset_dir):
        if filename.endswith(".png"):
            image_loc = osp.join(asset_dir, filename)
            img = Image.open(image_loc)
            mode = img.mode
            size = img.size
            data = img.tobytes()
            img_dict[image_loc] = (data, size, mode)

get_loaded_images(img_library)

