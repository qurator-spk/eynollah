import pytest
from PIL import Image

@pytest.mark.parametrize(
    "options",
    [
            [], # defaults
            ["-sos"],
    ], ids=str)
def test_run_eynollah_enhancement_filename(
    tmp_path,
    resources_dir,
    run_eynollah_ok_and_check_logs,
    options,
):
    infile = resources_dir / 'kant_aufklaerung_1784_0020.tif'
    outfile = tmp_path.joinpath('kant_aufklaerung_1784_0020.png')
    run_eynollah_ok_and_check_logs(
        'enhancement',
        [
            '-i', str(infile),
            '-o', str(outfile.parent),
        ] + options,
        [
            'Image was enhanced',
        ]
    )
    with Image.open(infile) as original_img:
        original_size = original_img.size
    with Image.open(outfile) as enhanced_img:
        enhanced_size = enhanced_img.size
    assert (original_size == enhanced_size) == ("-sos" in options)

def test_run_eynollah_enhancement_directory(
    tmp_path,
    resources_dir,
    image_resources,
    run_eynollah_ok_and_check_logs,
):
    outdir = tmp_path
    run_eynollah_ok_and_check_logs(
        'enhancement',
        [
            '-di', str(resources_dir),
            '-o', str(outdir),
        ],
        [
            f'Image {image_resources[0]} was enhanced',
            f'Image {image_resources[1]} was enhanced',
        ]
    )
    assert len(list(outdir.iterdir())) == 2
