from collections import namedtuple
import json
import os
from pathlib import Path
import time


class Seekable(object):
    '''
    A seekable file reader, writer which deals with newline delimited records. \n
    This reader maintains an index of line lengths, so seeking a line is a O(1) operation.
    '''

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
        has_newline = contents[-1] == '\n'
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
    
    def read_from(self, line_number):
        current_offset = self.file.tell()
        self.seek_line_start(line_number)
        lines = list()
        contents = self.readline()
        while len(contents) > 0:
            lines.append(contents)
            contents = self.readline()
        
        self.file.seek(current_offset)
        return lines
    
    def update_line(self, line_number, contents):
        lines = self.read_from(line_number)
        length = len(lines)
        self.truncate_until_end(line_number)
        self.writeline(contents)
        if length > 1:
            for line in lines[1:]:
                self.writeline(line)

    def lines(self):
        return len(self.line_lengths)

    def __exit__(self, type, value, traceback):
        self.file.close()


class Catalog(object):
    '''
    A new line delimited file that has records delimited by newlines. \n

    [ json object record ] \n
    [ json object record ] \n
    ...
    '''
    def __init__(self, path):
        self.path = Path(os.path.expanduser(path))
        self.seekable = Seekable(self.path.as_posix())


# Metadata for a catalog entry
CatalogMetadata = namedtuple('CatalogMetadata', 'path created_at start_index')


class Manifest(object):
    '''
    A newline delimited file, with the following format.

    [ json array of inputs ]\n
    [ json array of types ]\n
    [ json object with user metadata ]\n
    [ json object with manifest metadata ]\n
    '''

    def __init__(self, base_path, inputs=[], types=[], metadata=[], max_len=30000):
        self.base_path = Path(os.path.expanduser(base_path))
        self.manifest_path = Path(os.path.join(self.base_path, 'manifest.json'))
        self.inputs = inputs
        self.types = types
        self._read_metadata(metadata)
        self.manifest_metadata = dict()
        self.max_len = max_len
        self.seeker = Seekable(self.manifest_path)
        self.catalogs = list()

        if self.path.exists():
            self._read_contents()
        else:
            created_at = time.time()
            self.manifest_path.mkdir(parents=True, exist_ok=True)
            print('Created a new datastore at %s' % (self.base_path.as_posix()))

    def _read_metadata(self, metadata=[]):
        self.metadata = dict()
        for (key, value) in metadata:
            self.metadata[key] = value

    def _read_contents(self):
        self.seeker.seek_line_start(0)
        self.inputs = json.loads(self.seeker.readline())
        self.types = json.loads(self.seeker.readline())
        self.metadata = json.loads(self.seeker.readline())
        self.manifest_metadata = json.loads(self.seeker.readline())
        # Update catalog metadata

    def _write_contents(self):
        self.seeker.seek_line_start(0)
        self.seeker.writeline(json.dumps(self.inputs))
        self.seeker.writeline(json.dumps(self.types))
        self.seeker.writeline(json.dumps(self.metadata))
        self.seeker.writeline(json.dumps(self.manifest_metadata))

    def _update_manifest_metadata(self):
        self.seeker.truncate_until_end(3)
        self.seeker.writeline(json.dumps(self.manifest_metadata))
