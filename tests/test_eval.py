import unittest

import slope

class TestEval(unittest.TestCase):
    def test_add(self):
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)


if __name__ == "__main__":
    unittest.main()
