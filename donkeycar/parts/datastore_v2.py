from pathlib import Path
import os


class Seekable(object):
    def __init__(self, file, method='a+'):
        self.method = method
        self.line_lengths = list()
        self.file = open(file, self.method)
        self._read_contents()

    def _read_contents(self):
        self.line_lengths.clear()
        contents = self.file.readline()
        while len(contents) > 0:
            self.line_lengths.append(len(contents))
            contents = self.file.readline()
        self.seek_end_of_file()

    def __enter__(self):
        return self

    def writeline(self, contents):
        has_newline = '\n' in contents
        if has_newline:
            line = contents
        else:
            line = '%s\n' % (contents)

        offset = len(line)
        self.line_lengths.append(offset)
        self.file.write(line)
        self.file.flush()

    def _line_start_offset(self, line_number):
        return self._offset_until(line_number - 1)

    def _line_end_offset(self, line_number):
        return self._offset_until(line_number)

    def _offset_until(self, line_index):
        return sum(self.line_lengths[:line_index])

    def readline(self):
        return self.file.readline()

    def seek_line_start(self, line_number):
        self.file.seek(self._line_start_offset(line_number))

    def seek_end_of_file(self):
        self.file.seek(sum(self.line_lengths))

    def truncate_until_end(self, line_number):
        # Seek to end of previous line
        self.line_lengths = self.line_lengths[:line_number - 1]
        self.seek_end_of_file()
        self.file.truncate()

    def lines(self):
        return len(self.line_lengths)

    def __exit__(self, type, value, traceback):
        self.file.close()


class Manifest(object):
    def __init__(self, base_path, inputs=[], types=[], metadata=[]):
        self.path = Path(os.path.expanduser(base_path))
        self.inputs = inputs
        self.types = types
        self.metadata = metadata
        self.current_index = 0
