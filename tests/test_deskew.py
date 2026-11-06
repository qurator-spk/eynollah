import numpy as np

from eynollah.utils.rotate import rotate_image
from eynollah.utils.separate_lines import return_deskew_slop

def test_deskew_landscape_main_page():
    # landscape page (width > height) used to crash with
    # "TypeError: 'int' object is not iterable" (#223)
    img = np.zeros((80, 160))
    for y in (20, 40, 60):
        img[y: y + 3, 10: 150] = 1
    angle = return_deskew_slop(img, 1.0, None, n_tot_angles=10, main_page=True)
    assert angle is not None
    assert np.isclose(angle, 0, atol=0.02)

def test_deskew_rotated_rows():
    img = np.zeros((80, 160), dtype=np.uint8)
    for y in (20, 40, 60):
        img[y: y + 3, 10: 150] = 1
    img = rotate_image(img, 1.5)
    angle = return_deskew_slop(img, 1, None, n_tot_angles=10)
    assert angle is not None
    assert np.isclose(angle, -1.5, atol=0.02)

def test_deskew_rotated_cols():
    img = np.zeros((800, 640))
    for i in range(4):
        for j in range(30):
            y = j * 20 + 3 * i
            x = 160 * i
            img[y + 10: y + 21, x + 10: x + 150] = 1
    img = rotate_image(img, 1.5)
    angle = return_deskew_slop(img, 10, None, axis=0, n_tot_angles=10)
    assert angle is not None
    assert np.isclose(angle, -1.5, atol=0.02)
    angle2 = return_deskew_slop(img, 10, None, axis=(0, 1), n_tot_angles=10)
    assert angle is not None
    assert np.isclose(angle, -1.5, atol=0.02)
