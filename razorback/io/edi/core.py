""" load and write generic edi files

See: http://www.complete-mt-solutions.com/mtnet/docs/ediformat.txt
"""

import collections
import re
import io


__all__ = ['Block', 'load', 'write']


## TODO add sanity check to load()


Block = collections.namedtuple('Block', 'options dataset')
Block.__doc__ = """Block(options, dataset)

options: [dict] {name [str]: value [str]}
            the options of an edi block
            name and value are strings

dataset: [list of float]
            the dataset of an edi block
"""


def load(filename):
    """ load '.edi' file 

    return a dict of dict of list of Block

    keys of dicts are the keywords (str) of the sections and blocks 
    the unnamed sections and blocks are identified by the empty keywords: ''

    Format exception:
    As specified by the edi format, the special block 'INFO' of the unnamed
    section is raw text.
    Thus res['']['INFO'] is not a list of block but a string.

    A block has 2 fields: block.options and block.dataset
        block.options is a dict of name (str) to value (str)
        block.dataset is a list of float

    example:
        res = load('path/to/file.edi')
        res['']['INFO']                     -> raw text (str)
        res['']['HEAD'][0].options          -> options of HEAD section
        res['MTSECT']['FREQ'][0].dataset    -> the MT frequencies

    """

    p = {sec: ParserBase() for sec, _, _, _ in walk_file(open(filename))}
    p[''] = ParserMainSection()
    with open(filename) as f:
        parse(f, p)
    return {k: v.value for k, v in p.items()}


def walk_file(file):
    section = ''
    block = ''
    new_block = True
    for line in file:
        line = line.rstrip()
        if not line or line.startswith('>!'):
            continue
        if line.startswith('>='):
            new_block = True
            section = line[2:]
            block = ''
            continue
        if line.startswith('>'):
            new_block = True
            i = line.find(' ')
            if i < 0:
                block = line[1:]
                continue
            block = line[1:i]
            line = line[i+1:]
        yield section, block, new_block, line
        new_block = False


def parse(file, parsers):
    ignore = ParserIgnore()
    for section, block, new_block, line in walk_file(file):
        parsers.get(section, ignore).receive_line(line, block, new_block)
    for p in parsers.values():
        p.build()


class ParserIgnore():
    def build(self): pass
    def receive_line(self, line, block, new_block): pass


class ParserRawText():
    def __init__(self):
        self.value = collections.defaultdict(list)

    def build(self):
        self.value = {k: '\n'.join(v) for k, v in self.value.items()}

    def receive_line(self, line, block, new_block):
        self.value[block].append(line)


re_equal_pair = re.compile('([^\s=]+)\s*=\s*(?:"(.+?)"|([^\s=]+))')


class ParserBase():
    def __init__(self):
        self.value = None
        self._blocks = collections.defaultdict(list)
        self._in_dataset = False
        self._current_dataset_len = None

    def build(self):
        self.value = dict(self._blocks)

    def receive_line(self, line, block, new_block):
        if new_block:
            self._blocks[block].append(Block({}, []))
        
        if self._in_dataset:
            self.receive_dataset(block, line)
        elif '//' in line:
            before, _, after = line.partition('//')
            self.receive_line(before, block, False)
            self._in_dataset = True
            self.receive_line(after, block, False)
        else:
            options = re_equal_pair.findall(line)
            for key, val1, val2 in options:
                self._blocks[block][-1].options[key] = val1 or val2

    def receive_dataset(self, block, line):
        values = line.split()
        if self._current_dataset_len is None:
            self._current_dataset_len = int(values[0])
            values = values[1:]
        dataset = self._blocks[block][-1].dataset
        l = self._current_dataset_len
        values = values[:l-len(dataset)]
        dataset.extend(map(float, values))
        if len(dataset) >= l:
            self._in_dataset = False
            self._current_dataset_len = None


class ParserMainSection(ParserBase):
    def __init__(self, info_block='INFO'):
        super().__init__()
        self._info_parser = ParserRawText()
        self._info_block = info_block

    def build(self):
        super().build()
        self._info_parser.build()
        b = self._info_block
        self.value[b] = self._info_parser.value[b]

    def receive_line(self, line, block, new_block):
        if block == self._info_block:
            self._info_parser.receive_line(line, block, new_block)
        else:
            super().receive_line(line, block, new_block)


#############################################################################
#############################################################################


def write(filename, data):
    """ Write data into a EDI file.

    See load() a description of the data structure.
    """
    with open(filename, 'w') as f:
        dump(data, f)


class Streamer():
    def __init__(self, stream):
        self.stream = stream

    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.stream)


class DumperBase():
    def __init__(self, stream, *,
                 indent=2, section_mark='>=', block_mark='>', dataset_mark='//',
                 **kwargs):
        self.indent = ' ' * int(indent)
        self.section_mark = section_mark
        self.block_mark = block_mark
        self.dataset_mark = dataset_mark
        super().__init__(stream, **kwargs)

    def format_key(self, key):
        return str(key)

    def format_value(self, value):
        value = str(value)
        if ' ' in value:
            value = '"%s"' % value
        return value

    def section_title(self, name):
        return '%s%s' % (self.section_mark, name)

    def block_title(self, name):
        return '%s%s' % (self.block_mark, name)

    def represent(self, data):
        for nsec, section in data.items():
            self.represent_section(nsec, section)

    def represent_section(self, section_name, data):
        if section_name:
            self.print(self.section_title(section_name), sep='')
        for nblo, blocks in data.items():
            for block in blocks:
                self.represent_block(nblo, block)
        self.print()

    def represent_block(self, block_name, block):
        if block_name:
            self.print(self.block_title(block_name), sep='')
        self.represent_options(block.options)
        self.represent_dataset(block.dataset)

    def represent_options(self, options):
        for key, val in options.items():
            s = '%s=%s' % (self.format_key(key), self.format_value(val))
            self.print(self.indent, s, sep='')

    def represent_dataset(self, dataset):
        if not dataset:
            return
        self.print(self.indent, self.dataset_mark, ' ', len(dataset),
                   sep='', end='')
        for i, val in enumerate(dataset):
            if i % 6 == 0:
                self.print()
            self.print(' % .5e' % val, sep='', end='')
        self.print()


class DumperInfo():
    def represent_section(self, section_name, data):
        if section_name == '':
            self.represent_main_section(section_name, data)
        else:
            super().represent_section(section_name, data)

    def represent_main_section(self, section_name, data):
        if section_name:
            self.print(self.section_title(section_name), sep='')
        for nblo, blocks in data.items():
            if nblo == 'INFO':
                self.print(self.block_title(nblo))
                self.print(blocks)
                self.print()
                continue
            for block in blocks:
                self.represent_block(nblo, block)
                self.print()
        self.print()


class DumperOrdered():
    def __init__(
        self, stream, *,
        order = (('', ('HEAD', 'INFO', None)), ('DEFINEMEAS', ('', None)),
                 (None, ('', None))),
        **kwargs
    ):
        self.order = order
        super().__init__(stream, **kwargs)

    def key(self, sec_blo):
        order = [(s, b) for s, lb in self.order for b in lb]
        sec, blo = sec_blo
        for i, (o_sec, o_blo) in enumerate(order):
            if o_sec == sec or o_sec is None:
                if o_blo == blo or o_blo is None:
                    break
        return i, sec_blo

    def represent(self, data):
        sec_bl = sorted(get_structure(data), key=self.key)
        new_data = collections.OrderedDict()
        for sec, blo in sec_bl:
            new_data.setdefault(sec, {})[blo] = data[sec][blo]
        super().represent(new_data)


class Dumper(DumperOrdered, DumperInfo, DumperBase, Streamer):
    pass


def get_structure(data):
    return [(ns, nb) for ns, sec in data.items() for nb, _ in sec.items()]


def dump(data, stream=None, Dumper=Dumper, **kwargs):
    """ Serialize data into a EDI stream.

    If stream is None, return the produced string instead.
    kwargs are the option passed to Dumper for custom init.

    See load() to know the structure of data.
    """
    getvalue = None
    if stream is None:
        stream = io.StringIO()
        getvalue = stream.getvalue
    dumper = Dumper(stream, **kwargs)
    dumper.represent(data)
    if getvalue:
        return getvalue()
