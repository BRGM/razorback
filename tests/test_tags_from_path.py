import os

from razorback.utils import tags_from_path


def test_tags_from_path():
    tpl = os.path.normpath('path_{a}/to_{b}/my_{c}/file_{d}.txt')
    files = [tpl.format(a=_a, b=_b, c=_c, d=_d)
             for _a in 'AB'
             for _b in 'X'
             for _c in 'Y'
             for _d in '123'
            ]

    def test(pattern, *labels):
        tpl = "{0!r:<50}{1!r}"
        print()
        print(tpl.format(pattern, labels))
        for x, y in tags_from_path(files, pattern, *labels): print(tpl.format(x, y))

    test('path_{a}/to_{b}/my_{c}/file_{d}.txt', '{a}_{b}_{c}_{d}')
    test('path_{a}/to_{b}/my_{c}/file_{d}.txt', '{a}_{d}')
    test('path_{a}/to_{b}/my_{c}/file_{d}.txt', '{a}_{d}', '{a}')
    test('path_{a}/*/*/file_{d}.txt', '{a}_{d}')
    test('path_{a}/**/file_{d}.txt', '{a}_{d}')
    test('**_{d}.txt', '{d}')
    test('*[AB]/**/{i}.txt', '{i}')
    test('path_[!A]/**/{i}.txt', '{i}')

