import unittest

from squawkbox.commands.process_midi import MidiError, MidiObject, MidiHeader, MidiTrack


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