import pytest
from PIL import Image

@pytest.mark.parametrize(
    "options",
    [
            [], # defaults
            ["--no-patches"],
    ], ids=str)
def test_run_eynollah_binarization_filename(
    tmp_path,
    run_eynollah_ok_and_check_logs,
    resources_dir,
    options,
):
    infile = resources_dir + '/2files/kant_aufklaerung_1784_0020.tif'
    outfile = tmp_path / 'kant_aufklaerung_1784_0020.png'
    run_eynollah_ok_and_check_logs(
        'binarization',
        [
            '-i', str(infile),
            '-o', str(outfile),
        ] + options,
        [
            'Predicting'
        ]
    )
    assert outfile.exists()
    with Image.open(infile) as original_img:
        original_size = original_img.size
    with Image.open(outfile) as binarized_img:
        binarized_size = binarized_img.size
    assert original_size == binarized_size

def test_run_eynollah_binarization_directory(
    tmp_path,
    run_eynollah_ok_and_check_logs,
    resources_dir,
    image_resources,
):
    outdir = tmp_path
    run_eynollah_ok_and_check_logs(
        'binarization',
        [
            '-di', str(resources_dir / '2files'),
            '-o', str(outdir),
        ],
        [
            f'Predicting {image_resources[0].name}',
            f'Predicting {image_resources[1].name}',
        ]
    )
    assert len(list(outdir.iterdir())) == 2
