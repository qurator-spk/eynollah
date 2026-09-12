import pytest
from ocrd_modelfactory import page_from_file
from ocrd_models.constants import NAMESPACES as NS

@pytest.mark.parametrize(
    "options",
    [
            [], # defaults
            ["--model_based"],
    ], ids=str)
def test_run_eynollah_mbreorder_filename(
    tmp_path,
    resources_dir,
    run_eynollah_ok_and_check_logs,
    options,
):
    infile = resources_dir / '2files/kant_aufklaerung_1784_0020.xml'
    outfile = tmp_path /'kant_aufklaerung_1784_0020.xml'
    run_eynollah_ok_and_check_logs(
        'reorder',
        [
            '-i', str(infile),
            '-dim', str(resources_dir / '2files'),
            '-o', str(outfile.parent),
        ] + options,
        [
            str(infile)
        ]
    )
    assert outfile.exists()
    #in_tree = page_from_file(str(infile)).etree
    #in_order = in_tree.xpath("//page:OrderedGroup//@regionRef", namespaces=NS)
    out_tree = page_from_file(str(outfile)).etree
    out_order = out_tree.xpath("//page:OrderedGroup//@regionRef", namespaces=NS)
    #assert len(out_order) >= 2, "result is inaccurate"
    #assert in_order != out_order
    assert out_order == ['r_1_1', 'r_2_1', 'r_2_2', 'r_2_3']

def test_run_eynollah_mbreorder_directory(
    tmp_path,
    resources_dir,
    run_eynollah_ok_and_check_logs,
):
    outdir = tmp_path
    run_eynollah_ok_and_check_logs(
        'reorder',
        [
            '-di', str(resources_dir / '2files'),
            '-dim', str(resources_dir / '2files'),
            '-o', str(outdir),
        ],
        [
            'Job done in',
            'All jobs done in',
        ]
    )
    assert len(list(outdir.iterdir())) == 2

