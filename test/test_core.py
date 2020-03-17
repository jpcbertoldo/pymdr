from unittest import TestCase

from src import core


# noinspection PyArgumentList
class TestDataRegion(TestCase):
    def test__extra_format(self):
        dr = core.DataRegion(
            parent="body",
            gnode_size=3,
            first_gnode_start_index=5,
            n_nodes_covered=9,
        )
        dr._extra_format("!S")
        self.assertRaises(NotImplementedError, dr._extra_format, "!s")
        self.assertRaises(NotImplementedError, dr._extra_format, "d")

    def test_empty(self):
        self.fail()

    def test_binary_from_last_gnode(self):
        self.fail()

    def test_is_empty(self):
        self.fail()

    def test_n_gnodes(self):
        self.fail()

    def test_last_covered_tag_index(self):
        self.fail()

    def test_extend_one_gnode(self):
        self.fail()
