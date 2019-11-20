import os
import tempfile
import unittest

from donkeycar.parts.datastore_v2 import Seekable


class TestSeekeable(unittest.TestCase):

    def setUp(self):
        self._file, self._path = tempfile.mkstemp()

    def test_offset_tracking(self):
        appendable = Seekable(self._path)
        with appendable:
            appendable.writeline('Line 1')
            appendable.writeline('Line 2')
            self.assertEqual(len(appendable.line_lengths), 2)
            appendable.seek_line_start(1)
            self.assertEqual(appendable.readline(), 'Line 1\n')
            appendable.seek_line_start(2)
            self.assertEqual(appendable.readline(), 'Line 2\n')
            appendable.seek_end_of_file()
            appendable.truncate_until_end(2)
            appendable.writeline('Line 2 Revised')
            appendable.seek_line_start(2)
            self.assertEqual(appendable.readline(), 'Line 2 Revised\n')

    def test_read_contents(self):
        appendable = Seekable(self._path)
        with appendable:
            appendable.writeline('Line 1')
            appendable.writeline('Line 2')
            self.assertEqual(len(appendable.line_lengths), 2)
            appendable.file.seek(0)
            appendable._read_contents()
            self.assertEqual(len(appendable.line_lengths), 2)

    def tearDown(self):
        os.remove(self._path)


if __name__ == '__main__':
    unittest.main()
