import tempfile

from squawkbox.midi import Midi
from squawkbox.tokenizer import Tokenizer


def test_tokenize_and_detokenize():
    tokenizer = Tokenizer()
    with open('tests/fixtures/example.midi', 'rb') as midi_file:
        midi = Midi.load(midi_file)
        midi_file.seek(0)
        _bytes = midi_file.read()
    with open('tests/fixtures/example.txt', 'r') as token_file:
        tokens = token_file.read()
    assert tokenizer.tokenize(midi) == tokens
    assert tokenizer.detokenize(tokens) == _bytes
    assert tokenizer.detokenize(tokenizer.tokenize(midi)) == _bytes