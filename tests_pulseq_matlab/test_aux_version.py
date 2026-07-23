import pypulseq
from pypulseq_matlab_like.Sequence.sequence import Sequence


def _version_fields():
    major, minor, revision = (int(part) for part in pypulseq.__version__.split('.')[:3])
    return major, minor, revision, major * 1000000 + minor * 1000 + revision


class TestAuxVersion:
    def test_default_format(self):
        major, minor, revision, combined = _version_fields()
        assert major >= 1
        assert minor >= 5
        assert revision >= 0
        assert combined >= 1005001

    def test_output_struct(self):
        sequence = Sequence()
        major, minor, revision, combined = _version_fields()
        assert (sequence.version_major, sequence.version_minor, int(sequence.version_revision)) == (major, minor, revision)
        assert combined >= 1005001

    def test_combined(self):
        major, minor, revision, combined = _version_fields()
        assert combined == major * 1000000 + minor * 1000 + revision
