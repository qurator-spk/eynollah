import pytest
from ocrd_modelfactory import page_from_file
from ocrd_models.constants import NAMESPACES as NS

@pytest.mark.parametrize(
    "options",
    [
        [], # defaults
        ["-doit", #str(outrenderfile.parent)],
         ],
        ["-trocr"],
    ], ids=str)
def test_run_eynollah_ocr_filename(
    tmp_path,
    run_eynollah_ok_and_check_logs,
    resources_dir,
    options,
):
    infile = resources_dir / 'kant_aufklaerung_1784_0020.tif'
    outfile = tmp_path.joinpath('kant_aufklaerung_1784_0020.xml')
    outrenderfile = tmp_path / 'render' / 'kant_aufklaerung_1784_0020.png'
    outrenderfile.parent.mkdir()
    if "-doit" in options:
        options.insert(options.index("-doit") + 1, str(outrenderfile.parent))
    run_eynollah_ok_and_check_logs(
        'ocr',
        [
            '-i', str(infile),
            '-dx', str(infile.parent),
            '-o', str(outfile.parent),
        ] + options,
        [
            # FIXME: ocr has no logging!
        ]
    )
    assert outfile.exists()
    if "-doit" in options:
        assert outrenderfile.exists()
    #in_tree = page_from_file(str(infile)).etree
    #in_order = in_tree.xpath("//page:OrderedGroup//@regionRef", namespaces=NS)
    out_tree = page_from_file(str(outfile)).etree
    out_texts = out_tree.xpath("//page:TextLine/page:TextEquiv[last()]/page:Unicode/text()", namespaces=NS)
    assert len(out_texts) >= 2, ("result is inaccurate", out_texts)
    assert sum(map(len, out_texts)) > 100, ("result is inaccurate", out_texts)

def test_run_eynollah_ocr_directory(
    tmp_path,
    run_eynollah_ok_and_check_logs,
    resources_dir,
):
    outdir = tmp_path
    run_eynollah_ok_and_check_logs(
        'ocr',
        [
            '-di', str(resources_dir),
            '-dx', str(resources_dir),
            '-o', str(outdir),
        ],
        [
            # FIXME: ocr has no logging!
        ]
    )
    assert len(list(outdir.iterdir())) == 2

