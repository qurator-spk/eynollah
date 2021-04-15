import cv2
from pathlib import Path
from qurator.eynollah.utils.pil_cv2 import check_dpi
from tests.base import main

def test_dpi():
    fpath = str(Path(__file__).parent.joinpath('resources', 'kant_aufklaerung_1784_0020.tif'))
    assert 230 == check_dpi(cv2.imread(fpath))

if __name__ == '__main__':
    main(__file__)
