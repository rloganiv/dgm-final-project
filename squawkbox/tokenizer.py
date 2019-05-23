import logging

from squawkbox.midi import Midi

logger = logging.getLogger(__name__)


DEFAULT_TICKLEN = 1302
# TODO: Add configuration options.
class Tokenizer:

    def tokenize(self, midi: Midi) -> str:
        tokens = []

        ppqn = midi.header.pulses_per_quarter_note  # ticks per quarter note
        tempo = midi.tracks[0].events[0][1].metadata['tempo']  # micro-seconds per quarter note
        ticklen = tempo // ppqn  # micro-seconds per tick
        ticklen_mult = ticklen / DEFAULT_TICKLEN
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
        tokens = tokens.split()

