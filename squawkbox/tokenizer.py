from collections import deque
import logging

import squawkbox.midi as md

logger = logging.getLogger(__name__)


DEFAULT_TICKLEN = 1042
DEFAULT_HEADER = b'MThd\x00\x00\x00\x06\x00\x01\x00\x02\x01\xe0'
DEFAULT_TEMPO_MAP = b'MTrk\x00\x00\x00\x13\x00\xffQ\x03\x07\xa1 \x00\xffX\x04\x04\x02\x18\x08\x01\xff/\x00'

# TODO: Add configuration options.
class Tokenizer:

    def tokenize(self, midi: md.Midi) -> str:
        tokens = []

        ppqn = midi.header.pulses_per_quarter_note  # ticks per quarter note
        tempo = midi.tracks[0].events[0][1].metadata['tempo']  # micro-seconds per quarter note
        ticklen = tempo // ppqn  # micro-seconds per tick
        ticklen_mult = DEFAULT_TICKLEN / ticklen
        # bpm = 6e7 // tempo # quarter notes per minute
        # tokens.append(f'tempo:{int(bpm)}')
        # tokens.append(f'ticklen:{int(ticklen)}')
        tokens.append('start')

        cumulative_delta_time = 0
        for delta_time, event in midi.tracks[1].events:
            cumulative_delta_time += round(ticklen_mult * delta_time)
            if event.event_type == 'NoteOn':
                metadata = event.metadata
                if cumulative_delta_time > 0:
                    if tokens[-1] != 'start':
                        tokens.append(f'wait:{cumulative_delta_time}')
                    cumulative_delta_time = 0
                tokens.append(f'note:{metadata["key"]}:{metadata["velocity"]}')

        tokens.append('end')

        return ' '.join(tokens)

    def detokenize(self, tokens: str) -> str:
        tokens = deque(tokens.split())
        events = []
        delta_time = 0
        while tokens:
            current_token = tokens.popleft()
            token_type, *token_values = current_token.split(':')
            if token_type == 'wait':
                delta_time = int(token_values[0])
            elif token_type == 'note':
                metadata = {'key': int(token_values[0]), 'velocity': int(token_values[1])}
                event = md.MidiEvent(0x90, 'NoteOn', 0, metadata)
                events.append((delta_time, event))
                delta_time = 0
        track = md.MidiTrack(events)
        return DEFAULT_HEADER + DEFAULT_TEMPO_MAP + track.to_bytes()