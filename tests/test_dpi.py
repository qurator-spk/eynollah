from pathlib import Path
from sbb_newspapers_org_image.utils.pil_cv2 import check_dpi
from tests.base import main

def test_dpi():
    fpath = Path(__file__).parent.joinpath('resources', 'kant_aufklaerung_1784_0020.tif')
    assert 300 == check_dpi(str(fpath))

if __name__ == '__main__':
    main(__file__)
