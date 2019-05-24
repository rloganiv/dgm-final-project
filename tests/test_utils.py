from unittest import TestCase
from squawkbox.utils import Registrable


# Create a mock registrable object and register two subclasses, one which takes no args in
# its constructor and one which takes a single arg.
class Mock(Registrable):
    pass

@Mock.register('no-args')
class NoArgs(Mock):
    pass

@Mock.register('one-arg')
class OneArg(Mock):
    def __init__(self, arg):
        self.arg = arg


class TestRegistrable(TestCase):

    def test_get(self):
        mock_no_args = Mock.get('no-args')
        self.assertEqual(mock_no_args, NoArgs)

        mock_one_arg = Mock.get('one-arg')
        self.assertEqual(mock_one_arg, OneArg)

        with self.assertRaises(ValueError):
            Mock.get('does-not-exist')

    def test_from_config(self):
        no_args_config = {'name': 'no-args'}
        no_args = Mock.from_config(no_args_config)
        self.assertIsInstance(no_args, NoArgs)

        one_arg_config = {'name': 'one-arg', 'arg': 0}
        one_arg = Mock.from_config(one_arg_config)
        self.assertIsInstance(one_arg, OneArg)
        self.assertEqual(one_arg.arg, 0)
