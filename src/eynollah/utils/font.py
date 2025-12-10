
# cannot use importlib.resources until we move to 3.9+ forimportlib.resources.files
import sys
from PIL import ImageFont

if sys.version_info < (3, 10):
    import importlib_resources
else:
    import importlib.resources as importlib_resources


def get_font():
    #font_path = "Charis-7.000/Charis-Regular.ttf"  # Make sure this file exists!
    font = importlib_resources.files(__package__) / "../Charis-Regular.ttf"
    with importlib_resources.as_file(font) as font:
        return ImageFont.truetype(font=font, size=40)
