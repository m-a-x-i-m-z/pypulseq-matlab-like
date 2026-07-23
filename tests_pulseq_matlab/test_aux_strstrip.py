class TestAuxStrstrip:
    def test_leading_trailing_spaces(self):
        assert '  hello  '.strip() == 'hello'

    def test_no_whitespace(self):
        assert 'hello'.strip() == 'hello'

    def test_empty(self):
        assert ''.strip() == ''

    def test_only_spaces(self):
        assert '   '.strip() == ''

    def test_tabs_newlines(self):
        assert '\t hello \n'.strip() == 'hello'

    def test_internal_whitespace(self):
        assert '  hello world  '.strip() == 'hello world'
