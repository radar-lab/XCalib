"""The public names are lazy module attributes: dir() and help() must still list them."""

import xcalib


def test_dir_lists_every_public_name():
    assert set(xcalib.__all__) <= set(dir(xcalib))


def test_dir_is_sorted_and_has_no_duplicates():
    names = dir(xcalib)
    assert names == sorted(set(names))
