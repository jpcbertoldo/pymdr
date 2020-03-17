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
        dr = core.DataRegion.empty()
        self.assertIsNone(dr[0])
        self.assertEqual(dr.n_nodes_covered, 0)

    def test_binary_from_last_gnode(self):
        gn = core.GNode("table-0", 4, 6)
        dr = core.DataRegion.binary_from_last_gnode(gn)
        self.assertEqual(dr.parent, gn.parent)
        self.assertEqual(dr.gnode_size, gn.size)
        self.assertEqual(dr.first_gnode_start_index, gn.start - gn.size)
        self.assertEqual(dr.n_nodes_covered, 2 * gn.size)
        self.assertEqual(dr.last_covered_tag_index, gn.end - 1)
        self.assertEqual(dr.n_gnodes, 2)
        self.assertTrue(4 in dr)
        self.assertTrue(5 in dr)
        self.assertFalse(6 in dr)

    def test_is_empty(self):
        empty = core.DataRegion.empty()
        non_empty = core.DataRegion("table-0", 1, 5, 3)
        self.assertTrue(empty.is_empty)
        self.assertFalse(non_empty.is_empty)

    def test_n_gnodes(self):
        dr = core.DataRegion("tr-9", 2, 0, 2 * 3)
        self.assertEqual(dr.n_gnodes, 3)

    def test_last_covered_tag_index(self):
        dr = core.DataRegion("tr-9", 2, 0, 2 * 3)
        self.assertEqual(dr.last_covered_tag_index, 5)

    def test_extend_one_gnode(self):
        dr = core.DataRegion("tr-9", 2, 0, 2 * 3)
        dr_ext = dr.extend_one_gnode()
        self.assertEqual(dr_ext, core.DataRegion("tr-9", 2, 0, 2 * 4))

    def test_dunders(self):
        dr = core.DataRegion("tr-9", 2, 5, 2 * 2)
        "{}".format(dr)
        "{:!s}".format(dr)
        "{:!S}".format(dr)
        "{:!r}".format(dr)
        self.assertTrue(4 not in dr)
        self.assertTrue(9 not in dr)
        self.assertTrue(all(i in dr for i in range(5, 9)))
        gnodes = list(dr)
        self.assertEqual(len(gnodes), 2)
        self.assertEqual(gnodes[0], core.GNode("tr-9", 5, 7))
        self.assertEqual(gnodes[1], core.GNode("tr-9", 7, 9))
