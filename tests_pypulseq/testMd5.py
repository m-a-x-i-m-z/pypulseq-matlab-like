import hashlib


def _md5(value):
    return hashlib.md5(value.encode('ascii')).hexdigest()


class TestMd5:
    def test_empty_string(self):
        assert _md5('') == 'd41d8cd98f00b204e9800998ecf8427e'

    def test_single_char_a(self):
        assert _md5('a') == '0cc175b9c0f1b6a831c399e269772661'

    def test_abc(self):
        assert _md5('abc') == '900150983cd24fb0d6963f7d28e17f72'

    def test_message_digest(self):
        assert _md5('message digest') == 'f96b697d7cb7938d525a2f31aaf161d0'

    def test_quick_brown_fox(self):
        assert _md5('The quick brown fox jumps over the lazy dog') == '9e107d9d372bb6826bd81d3542a419d6'

    def test_alphabet(self):
        assert _md5('abcdefghijklmnopqrstuvwxyz') == 'c3fcd3d76192e4007dfb496cca67e13b'

    def test_source(self):
        assert _md5('abcdefghijklmnopqrstuvwxyz') == 'c3fcd3d76192e4007dfb496cca67e13b'

    def test_determinism(self):
        assert _md5('hello world') == _md5('hello world')

    def test_transposed_input(self):
        assert _md5('hello world') == _md5('hello world')
