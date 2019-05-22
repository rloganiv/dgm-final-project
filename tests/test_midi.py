from collections import deque
import unittest

import squawkbox.midi as midi


def test_parse_variable_length_quantity():
    # Single byte
    byte_string = b'\x7F'
    expected = 127
    byte_queue = deque(byte_string)
    observed = midi._parse_variable_length_quantity(byte_queue)
    assert expected == observed

    # Multiple bytes
    byte_string = b'\x81\x00'
    expected = 128
    byte_queue = deque(byte_string)
    observed = midi._parse_variable_length_quantity(byte_queue)
    assert expected == observed

    byte_string = b'\xBD\x84\x40'
    expected = 1000000
    byte_queue = deque(byte_string)
    observed = midi._parse_variable_length_quantity(byte_queue)
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
        midi_header = midi.MidiHeader.from_bytes(good_combination)
        self.assertEqual(midi_header.format_type, 1)
        self.assertEqual(midi_header.ntracks, 2)
        self.assertEqual(midi_header.pulses_per_quarter_note, 255)

    def test_unsupported(self):
        # Check that errors are raised on unsupported headers.
        bad_tickdiv_type = self.format_type + self.ntracks + self.tickdiv_type_1
        with self.assertRaises(NotImplementedError):
            midi_header = midi.MidiHeader.from_bytes(bad_tickdiv_type)


class TestMidiTrack(unittest.TestCase):
    def test_from_bytes(self):
        # Check error raised on invalid prefix
        invalid_prefix = b'\x00\x00'
        with self.assertRaises(midi.MidiError):
            midi_track = midi.MidiTrack.from_bytes(invalid_prefix)

        # Check can read a SysexEvent
        sysex_event = b'\x00\xF0\x01\x00'
        midi_track = midi.MidiTrack.from_bytes(sysex_event)
        self.assertEqual(len(midi_track.events), 1)

        delta_time, event = midi_track.events[0]
        self.assertEqual(delta_time, 0)
        self.assertIsInstance(event, midi.SysexEvent)
        self.assertEqual(event.metadata['raw_data'], b'')

        # Check can read a MetaEvent
        meta_event = b'\x00\xFF\x51\x03\x00\x00\x00'
        midi_track = midi.MidiTrack.from_bytes(meta_event)
        self.assertEqual(len(midi_track.events), 1)

        _, event = midi_track.events[0]
        self.assertIsInstance(event, midi.MetaEvent)
        self.assertEqual(event.event_type, 'SetTempo')
        self.assertEqual(event.metadata['tempo'], 0)

        # Check can read two MIDI events (so we can evaluate running status)
        midi_events = b'\x00\x90\x0F\x0F\x00\x00\x00'
        midi_track = midi.MidiTrack.from_bytes(midi_events)
        self.assertEqual(len(midi_track.events), 2)

        _, event = midi_track.events[0]
        self.assertIsInstance(event, midi.MidiEvent)
        self.assertEqual(event.event_type, 'NoteOn')
        self.assertEqual(event.metadata['key'], 15)
        self.assertEqual(event.metadata['velocity'], 15)

        _, event = midi_track.events[1]
        self.assertIsInstance(event, midi.MidiEvent)
        self.assertEqual(event.event_type, 'NoteOn')
        self.assertEqual(event.metadata['key'], 0)
        self.assertEqual(event.metadata['velocity'], 0)


class TestMidiFile(unittest.TestCase):
    def test_loads(self):
        # Check that no errors occur when loading a MIDI file.
        with open('tests/fixtures/example.midi', 'rb') as f:
            midi.Midi.load(f)
