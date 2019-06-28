from collections import deque
import logging

import squawkbox.midi as md

logger = logging.getLogger(__name__)


DEFAULT_TICKLEN = 500000 / 480
DEFAULT_HEADER = b'MThd\x00\x00\x00\x06\x00\x01\x00\x02\x01\xe0'
DEFAULT_TEMPO_MAP = b'MTrk\x00\x00\x00\x13\x00\xffQ\x03\x07\xa1 \x00\xffX\x04\x04\x02\x18\x08\x01\xff/\x00'


def split_waits(delta_time):
    n_waits = delta_time // 100
    out = ['wait:100'] * n_waits
    remainder = delta_time % 100
    if remainder > 0:
        out.append(f'wait:{remainder}')
    return out


# TODO: Add configuration options.
class Tokenizer:

    def __init__(self, scale=1, max_tokens=None, max_wait_time=None):
        self._scale = scale
        self._max_tokens = max_tokens
        self._max_wait_time = max_wait_time

    def tokenize(self, midi: md.Midi) -> str:
        tokens = []

        ppqn = midi.header.pulses_per_quarter_note  # ticks per quarter note
        logger.debug('pulses per quarter note: %i', ppqn)
        tempo = midi.tracks[0].events[0][1].metadata['tempo']  # micro-seconds per quarter note
        logger.debug('tempo: %i', tempo)
        ticklen = tempo / ppqn  # micro-seconds per tick
        logger.debug('ticklen: %f', ticklen)
        ticklen_mult = DEFAULT_TICKLEN / ticklen
        # bpm = 6e7 // tempo # quarter notes per minute
        # tokens.append(f'tempo:{int(bpm)}')
        # tokens.append(f'ticklen:{int(ticklen)}')
        tokens.append('start')

        cumulative_delta_time = 0
        for i, (delta_time, event) in enumerate(midi.tracks[1].events):
            if self._max_tokens is not None:
                if len(tokens) >= self._max_tokens:
                    yield ' '.join(tokens[:self._max_tokens])
                    tokens = ['continue', *tokens[self._max_tokens:]]
            cumulative_delta_time += round(delta_time / ticklen_mult)
            if event.event_type == 'NoteOn':
                metadata = event.metadata
                if cumulative_delta_time > 0:
                    if tokens[-1] != 'start':
                        tokens.extend(split_waits(cumulative_delta_time))
                    cumulative_delta_time = 0
                if metadata['velocity'] > 0:
                    velocity = 60
                else:
                    velocity = 0
                tokens.append(f'note:{metadata["key"]}:{velocity}')

        tokens.append('end')
        if self._max_tokens is not None:
            if len(tokens) >= self._max_tokens:
                yield ' '.join(tokens[:self._max_tokens])
                tokens = tokens[self._max_tokens:]
        yield ' '.join(tokens)

    def detokenize(self, tokens: str) -> str:
        tokens = deque(tokens.split())
        events = []
        delta_time = 0
        while tokens:
            current_token = tokens.popleft()
            token_type, *token_values = current_token.split(':')
            if token_type == 'wait':
                delta_time += self._scale * int(token_values[0])
            elif token_type == 'note':
                metadata = {'key': int(token_values[0]), 'velocity': int(token_values[1])}
                event = md.MidiEvent(0x90, 'NoteOn', 0, metadata)
                events.append((delta_time, event))
                delta_time = 0
        track = md.MidiTrack(events)
        return DEFAULT_HEADER + DEFAULT_TEMPO_MAP + track.to_bytes()

