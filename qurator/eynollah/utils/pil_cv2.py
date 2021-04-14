from PIL import Image
import numpy as np
from ocrd_models import OcrdExif
from cv2 import COLOR_GRAY2BGR, COLOR_RGB2BGR, cvtColor, imread

# from sbb_binarization

def cv2pil(img):
    return Image.fromarray(img.astype('uint8'))

def pil2cv(img):
    # from ocrd/workspace.py
    color_conversion = COLOR_GRAY2BGR if img.mode in ('1', 'L') else  COLOR_RGB2BGR
    pil_as_np_array = np.array(img).astype('uint8') if img.mode == '1' else np.array(img)
    return cvtColor(pil_as_np_array, color_conversion)

def check_dpi(image_filename):
    try:
        exif = OcrdExif(Image.open(image_filename))
        resolution = exif.resolution
        if resolution == 1:
            raise Exception()
        if exif.resolutionUnit == 'cm':
            resolution /= 2.54
        return int(resolution)
    except:
        return 230
