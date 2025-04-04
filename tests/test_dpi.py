import cv2
from pathlib import Path
from eynollah.utils.pil_cv2 import check_dpi

def test_dpi():
    fpath = str(Path(__file__).parent.joinpath('resources', 'kant_aufklaerung_1784_0020.tif'))
    assert 230 == check_dpi(cv2.imread(fpath))

