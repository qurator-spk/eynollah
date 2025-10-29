import pytest
from ocrd_modelfactory import page_from_file
from ocrd_models.constants import NAMESPACES as NS

@pytest.mark.parametrize(
    "options",
    [
            [], # defaults
            #["--allow_scaling", "--curved-line"],
            ["--allow_scaling", "--curved-line", "--full-layout"],
            ["--allow_scaling", "--curved-line", "--full-layout", "--reading_order_machine_based"],
            ["--allow_scaling", "--curved-line", "--full-layout", "--reading_order_machine_based",
             "--textline_light", "--light_version"],
            # -ep ...
            # -eoi ...
            # FIXME: find out whether OCR extra was installed, otherwise skip these
            ["--do_ocr"],
            ["--do_ocr", "--light_version", "--textline_light"],
            ["--do_ocr", "--transformer_ocr"],
            #["--do_ocr", "--transformer_ocr", "--light_version", "--textline_light"],
            ["--do_ocr", "--transformer_ocr", "--light_version", "--textline_light", "--full-layout"],
            # --skip_layout_and_reading_order
    ], ids=str)
def test_run_eynollah_layout_filename(
    tmp_path,
    run_eynollah_ok_and_check_logs,
    resources_dir,
    options,
):
    infile = resources_dir / 'kant_aufklaerung_1784_0020.tif'
    outfile = tmp_path / 'kant_aufklaerung_1784_0020.xml'
    run_eynollah_ok_and_check_logs(
        'layout',
        [
        '-i', str(infile),
        '-o', str(outfile.parent),
        ] + options,
        [
            str(infile)
        ]
    )
    assert outfile.exists()
    tree = page_from_file(str(outfile)).etree
    regions = tree.xpath("//page:TextRegion", namespaces=NS)
    assert len(regions) >= 2, "result is inaccurate"
    regions = tree.xpath("//page:SeparatorRegion", namespaces=NS)
    assert len(regions) >= 2, "result is inaccurate"
    lines = tree.xpath("//page:TextLine", namespaces=NS)
    assert len(lines) == 31, "result is inaccurate" # 29 paragraph lines, 1 page and 1 catch-word line

@pytest.mark.parametrize(
    "options",
    [
            ["--tables"],
            ["--tables", "--full-layout"],
            ["--tables", "--full-layout", "--textline_light", "--light_version"],
    ], ids=str)
def test_run_eynollah_layout_filename2(
    tmp_path,
    resources_dir,
    run_eynollah_ok_and_check_logs,
    options,
):
    infile = resources_dir / 'euler_rechenkunst01_1738_0025.tif'
    outfile = tmp_path / 'euler_rechenkunst01_1738_0025.xml'
    run_eynollah_ok_and_check_logs(
        'layout',
        [
            '-i', str(infile),
            '-o', str(outfile.parent),
        ] + options,
        [
            infile
        ]
    )
    assert outfile.exists()
    tree = page_from_file(str(outfile)).etree
    regions = tree.xpath("//page:TextRegion", namespaces=NS)
    assert len(regions) >= 2, "result is inaccurate"
    regions = tree.xpath("//page:TableRegion", namespaces=NS)
    # model/decoding is not very precise, so (depending on mode) we can get fractures/splits/FP
    assert len(regions) >= 1, "result is inaccurate"
    regions = tree.xpath("//page:SeparatorRegion", namespaces=NS)
    assert len(regions) >= 2, "result is inaccurate"
    lines = tree.xpath("//page:TextLine", namespaces=NS)
    assert len(lines) >= 2, "result is inaccurate" # mostly table (if detected correctly), but 1 page and 1 catch-word line

def test_run_eynollah_layout_directory(
    tmp_path,
    resources_dir,
    run_eynollah_ok_and_check_logs,
):
    outdir = tmp_path
    run_eynollah_ok_and_check_logs(
        'layout',
        [
        '-di', str(resources_dir),
        '-o', str(outdir),
        ],
        [
            'Job done in',
            'All jobs done in',
        ]
    )
    assert len(list(outdir.iterdir())) == 2
