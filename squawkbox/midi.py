"""
MIDI objects
"""
from collections import deque
import logging
from typing import Any, Deque, Dict, IO, List, Tuple

from overrides import overrides


logger = logging.getLogger(__name__)


HEADER_IDENTIFIER = b'MThd'
TRACK_IDENTIFIER = b'MTrk'


def _parse_variable_length_quantity(byte_queue: Deque[int]) -> int:
    """
    Parses a variable length quantity. The most significant bit indicates whether or not to
    continue accumulating, and the remaining 7 bits store the quantity to accumulate.
    """
    quantity = 0
    while True:
        byte = byte_queue.popleft()
        quantity = (quantity << 7) + (byte & 0x7F)
        if not (byte >> 7) & 1:
            break
    return quantity


def _pop_bytes(byte_queue: Deque[int], n: int) -> bytes:
    return bytes(byte_queue.popleft() for _ in range(n))


class MidiError(Exception): pass


class MidiFile:
    """
    An object for storing parsed MIDI information.
    """
    def __init__(self,
                 header: 'MidiHeader',
                 tracks: List['MidiTrack']) -> None:
        self.header = header
        self.tracks = tracks

    @classmethod
    def load(cls, f) -> 'MidiFile':
        tracks: List['MidiTrack'] = []
        while True:
            identifier = f.read(4)
            if identifier == b'':
                break
            chunklen = int.from_bytes(f.read(4), byteorder='big')
            chunk = f.read(chunklen)
            if identifier == HEADER_IDENTIFIER:
                header = MidiHeader.from_bytes(chunk)
                logger.debug(f'Parsed Header: {header}')
            elif identifier == TRACK_IDENTIFIER:
                midi_track = MidiTrack.from_bytes(chunk)
                tracks.append(midi_track)
            else:
                MidiError(f'Encountered unknown identifier "{identifier}".')
        return cls(header, tracks)


class MidiHeader:
    """
    An object for reading/storing information from the header chunk in a MIDI file.

    TODO: Support frame-based tickdiv.

    Parameters
    ==========
    format_type : ``int``
        The format type of the MIDI file. Must be one of: 0, 1, 2.
    ntracks : ``int``
        The number of tracks in the MIDI file.
    pulses_per_quarter_note : ``int``
        The number of delta time units in each quarter note.
    """
    def __init__(self,
                 format_type: int,
                 ntracks: int,
                 pulses_per_quarter_note: int) -> None:
        self.format_type = format_type
        self.ntracks = ntracks
        self.pulses_per_quarter_note = pulses_per_quarter_note

    def __repr__(self) -> str:
        return (f'MidiHeader(format_type={self.format_type}, '
                           f'ntracks={self.ntracks}, '
                           f'ppqn={self.pulses_per_quarter_note})')

    @classmethod
    def from_bytes(cls, chunk: bytes) -> 'MidiHeader':
        format_type = int.from_bytes(chunk[0:2], byteorder='big')
        ntracks = int.from_bytes(chunk[2:4], byteorder='big')
        tickdiv = int.from_bytes(chunk[4:6], byteorder='big')

        # TODO: Handle timecode timining intervals
        timing_interval = (tickdiv >> 15) & 1
        if timing_interval == 1:
            raise NotImplementedError('Timecode timing intervals not supported.')
            # remainder = tickdiv & 0x7fff  # Mask first bit
        pulses_per_quarter_note = tickdiv

        return cls(format_type, ntracks, pulses_per_quarter_note)


class MidiTrack:
    """
    An object for reading/storing information from a track chunk in a MIDI file.
    """
    def __init__(self, events: List[Tuple[int, 'Event']]) -> None:
        self.events = events

    @classmethod
    def from_bytes(cls,
                   chunk: bytes) -> 'MidiTrack':
        byte_queue = deque(chunk)
        event = None
        events = []
        event = None
        while len(byte_queue) > 0:
            delta_time = _parse_variable_length_quantity(byte_queue)
            event = MidiTrack._parse_event(byte_queue, event)
            logger.debug(f'Delta = {delta_time}, Event = {event}')
            events.append((delta_time, event))
        return cls(events)

    @staticmethod
    def _parse_event(byte_queue: Deque[int], prev_event: 'Event' = None) -> 'Event':
        prefix = byte_queue.popleft()
        if prefix in SysexEvent.PREFIXES:
            return SysexEvent.from_byte_queue(byte_queue, prefix)
        elif prefix in MetaEvent.PREFIXES:
            return MetaEvent.from_byte_queue(byte_queue, prefix)
        elif (prefix >> 4) in MidiEvent.PREFIXES:
            return MidiEvent.from_byte_queue(byte_queue, prefix)
        elif (prefix < 0x80) and isinstance(prev_event, MidiEvent):
            byte_queue.appendleft(prefix)
            return MidiEvent.from_byte_queue(byte_queue, prev_event.prefix)
        else:
            raise MidiError(f'Encountered unknown event prefix "{prefix:02x}".')


class Event:
    def __repr__(self):
        args = ', '.join('%s=%s' % item for item in self.__dict__.items())
        return f'{self.__class__.__name__}({args})'

    @classmethod
    def from_byte_queue(cls, byte_queue: Deque[int], prefix: int) -> 'Event':
        raise NotImplementedError('Method must be overridden by subclasses.')


class SysexEvent(Event):
    """
    A Sysex event.

    Don't know what these do, or really care...
    """
    PREFIXES = {0xF0, 0xF7}
    def __init__(self, prefix: int, metadata: Dict[str, Any]) -> None:
        self.prefix = prefix
        self.event_type = 'SysEx'
        self.metadata = metadata

    @classmethod
    @overrides
    def from_byte_queue(cls, byte_queue: Deque[int], prefix: int) -> 'SysexEvent':
        length = _parse_variable_length_quantity(byte_queue)
        metadata: Dict[str, Any] = {'raw_data': b''.join(bytes(byte_queue.popleft()) for _ in range(length))}
        return cls(prefix, metadata)


class MetaEvent(Event):
    """
    A meta event (e.g. tempo, time signature, etc.).

    Parameters
    ==========
    event_type : ``str``
        The type of event being triggered.

    NOTE: In addition to the parameters above, assorted message parameters may also be assigned.
    """
    PREFIXES = {0xFF}
    EVENT_TYPES = {
        0x00: 'SequenceNumber',
        0x01: 'Text',
        0x02: 'Copyright',
        0x03: 'SequenceName',
        0x04: 'InstrumentName',
        0x05: 'Lyric',
        0x06: 'Marker',
        0x07: 'CuePoint',
        0x20: 'ChannelPrefix',
        0x2F: 'EndOfTrack',
        0x51: 'SetTempo',
        0x54: 'SmtpeOffset',
        0x58: 'TimeSignature',
        0x59: 'KeySignature'
    }
    def __init__(self, prefix: int, event_type: str, metadata: Dict[str, Any]) -> None:
        self.prefix = prefix
        self.event_type = event_type
        self.metadata = metadata

    @classmethod
    @overrides
    def from_byte_queue(cls, byte_queue: Deque[int], prefix: int) -> 'MetaEvent':
        event_byte = byte_queue.popleft()
        if event_byte not in MetaEvent.EVENT_TYPES:
            raise MidiError(f'Unknown meta event type "{event_byte:02x}".')
        event_type = MetaEvent.EVENT_TYPES[event_byte]

        metadata: Dict[str, Any] = {}
        if event_type in ['Text', 'Copyright', 'SequenceName', 'InstrumentName', 'Lyric', 'Marker', 'CuePoint']:
            length = byte_queue.popleft()
            metadata['text'] = _pop_bytes(byte_queue, length).decode('ascii')
        if event_type == 'SetTempo':
            byte_queue.popleft()
            metadata['tempo'] = int.from_bytes(_pop_bytes(byte_queue, 3), byteorder='big')
        if event_type == 'TimeSignature':
            byte_queue.popleft()
            numerator = byte_queue.popleft()
            denominator = 2**byte_queue.popleft()
            metadata['time_signature'] = f'{numerator}/{denominator}'
            metadata['raw_data'] = _pop_bytes(byte_queue, 2)
        if event_type == 'KeySignature':
            byte_queue.popleft()
            metadata['offset'] = int.from_bytes(bytes(byte_queue.popleft()), byteorder='big', signed=True)
            metadata['tonality'] = 'minor' if byte_queue.popleft() else 'major'
        if event_type == 'SequenceNumber':
            metadata['raw_data'] = _pop_bytes(byte_queue, 3)
        if event_type == 'ChannelPrefix':
            metadata['raw_data'] = _pop_bytes(byte_queue, 2)
        if event_type == 'EndOfTrack':
            byte_queue.popleft()
        if event_type == 'SmtpeOffset':
            metadata['raw_data'] = _pop_bytes(byte_queue, 6)

        return cls(prefix, event_type, metadata)


class MidiEvent(Event):
    """
    A MIDI event (e.g. 'NoteOn', 'NoteOff', etc.).

    Parameters
    ==========
    event_type : ``str``
        The type of event being triggered.
    channel : ``int``
        The target MIDI channel.

    NOTE: In addition to the parameters above, assorted message parameters may also be assigned.
    """
    PREFIXES = {0x8,0x9, 0xA, 0xB, 0xC, 0xD, 0xE}
    EVENT_TYPES = {
        0x8: 'NoteOff',
        0x9: 'NoteOn',
        0XA: 'Aftertouch',
        0xB: 'Controller',
        0xC: 'Program',
        0xD: 'ChannelKeyPressure',
        0xE: 'PitchBend'
    }
    def __init__(self,
                 prefix: int,
                 event_type: str,
                 channel: int,
                 metadata: Dict[str, Any]) -> None:
        self.prefix = prefix
        self.event_type = event_type
        self.channel = channel
        self.metadata = metadata

    @classmethod
    @overrides
    def from_byte_queue(cls, byte_queue: Deque[int], prefix: int) -> 'MidiEvent':
        event_type = MidiEvent.EVENT_TYPES[prefix >> 4]
        channel = prefix & 0x0F
        metadata: Dict[str, Any] = {}
        if event_type in ['NoteOff', 'NoteOn']:
            metadata['key'] = byte_queue.popleft()
            metadata['velocity'] = byte_queue.popleft()
        elif event_type in ['Program', 'ChannelKeyPressure']:
            metadata['raw_data'] = _pop_bytes(byte_queue, 1)
        else:
            metadata['raw_data'] = _pop_bytes(byte_queue, 2)

        return cls(prefix, event_type, channel, metadata)
