import unittest

import src.utils as utils


class TestUtils(unittest.TestCase):
    def test_generate_random_colors(self):
        colors = utils.generate_random_colors(3)
        self.assertEquals(len(colors), 3)
        self.assertTrue(
            all(c1 != c2 for c1, c2 in zip(colors[:-1], colors[1:]))
        )
