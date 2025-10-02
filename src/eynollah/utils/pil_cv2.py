from contextlib import nullcontext
from PIL import Image
import numpy as np
from ocrd_models import OcrdExif
from cv2 import COLOR_GRAY2BGR, COLOR_RGB2BGR, COLOR_BGR2RGB, cvtColor, imread

# from sbb_binarization

def cv2pil(img):
    # reduce depth because cvtColor is limited
    return Image.fromarray(np.array(cvtColor(img.astype(np.uint8), COLOR_BGR2RGB)))

def pil2cv(img):
    # from ocrd/workspace.py
    color_conversion = COLOR_GRAY2BGR if img.mode in ('1', 'L', 'LA') else  COLOR_RGB2BGR
    pil_as_np_array = np.array(img).astype('uint8') if img.mode == '1' else np.array(img)
    # cvtColor cannot handle alpha
    if pil_as_np_array.shape[-1] == 2:
        pil_as_np_array = pil_as_np_array[:,:,0]
    elif pil_as_np_array.shape[-1] == 4:
        pil_as_np_array = pil_as_np_array[:,:,:3]
    return cvtColor(pil_as_np_array, color_conversion)

def check_dpi(img):
    try:
        if isinstance(img, Image.Image):
            pil_image = nullcontext(img)
        elif isinstance(img, str):
            pil_image = Image.open(img)
        else:
            pil_image = nullcontext(cv2pil(img))
        with pil_image:
            exif = OcrdExif(pil_image)
        resolution = exif.resolution
        if resolution == 1:
            raise Exception()
        if exif.resolutionUnit == 'cm':
            resolution /= 2.54
        return int(resolution)
    except Exception as e:
        print(e)
        return 230
