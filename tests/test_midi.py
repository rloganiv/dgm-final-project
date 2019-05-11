from collections import deque
import unittest

from squawkbox.midi import _parse_variable_length_quantity, MidiError, MidiHeader, MidiTrack


def test_parse_variable_length_quantity():
    # Single byte
    byte_string = 0x7F.to_bytes(1, 'big')
    expected = 127
    byte_queue = deque(byte_string)
    observed = _parse_variable_length_quantity(byte_queue)
    assert expected == observed

    # Multiple bytes
    byte_string = 0x8100.to_bytes(2, 'big')
    expected = 128
    byte_queue = deque(byte_string)
    observed = _parse_variable_length_quantity(byte_queue)
    assert expected == observed

    byte_string = 0xBD8440.to_bytes(3, 'big')
    expected = 1000000
    byte_queue = deque(byte_string)
    observed = _parse_variable_length_quantity(byte_queue)
    assert expected == observed


class TestMidiHeader(unittest.TestCase):
    def setUp(self):
        self.format_type = b'\x00\x01'
        self.ntracks = b'\x00\x02'
        self.tickdiv_type_0 = b'\x00\xFF'
        self.tickdiv_type_1 = b'\x80\x00'

    def test_from_bytes(self):
        # Check that a well-formed header chunk is properly read.
        good_combination = self.format_type + self.ntracks + self.tickdiv_type_0
        header = MidiHeader.from_bytes(good_combination)
        self.assertEqual(header.format_type, 1)
        self.assertEqual(header.ntracks, 2)
        self.assertEqual(header.pulses_per_quarter_note, 255)

    def test_unsupported(self):
        # Check that errors are raised on unsupported headers.
        bad_tickdiv_type = self.format_type + self.ntracks + self.tickdiv_type_1
        with self.assertRaises(NotImplementedError):
            header = MidiHeader.from_bytes(bad_tickdiv_type)