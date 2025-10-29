from ocrd_modelfactory import page_from_file
from ocrd_models.constants import NAMESPACES as NS

from eynollah.cli import (
    machine_based_reading_order as mbreorder_cli,
)


def test_run_eynollah_mbreorder_filename(
    tmp_path,
    resources_dir,
    run_eynollah_ok_and_check_logs,
):
    infile = resources_dir / 'kant_aufklaerung_1784_0020.xml'
    outfile = tmp_path.joinpath('kant_aufklaerung_1784_0020.xml')
    run_eynollah_ok_and_check_logs(
        mbreorder_cli,
        [
            '-i', str(infile),
            '-o', str(outfile.parent),
        ],
        [
            # FIXME: mbreorder has no logging!
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
    outfile = tmp_path.joinpath('kant_aufklaerung_1784_0020.xml')
    outdir = tmp_path
    run_eynollah_ok_and_check_logs(
        mbreorder_cli,
        [
            '-di', str(resources_dir),
            '-o', str(outdir),
        ],
        [
            # FIXME: mbreorder has no logging!
        ]
    )
    assert len(list(outdir.iterdir())) == 2

