from random import randint
from random import uniform

from squawkbox.utils import Registrable


# Create a new registrable class
class Transform(Registrable): pass


# Stretches wait time durations
@Transform.register("time-stretch")
class TimeStretch(Transform):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, tokens):
        k = uniform(self.min, self.max)
        new_tokens = []
        for token in tokens:
            token_type, *values = token.split(':')
            if token_type == 'wait':
                wait_time = int(values[0])
                new_token = 'wait:%d' % (min(4095, round(wait_time * k)))
                new_tokens.append(new_token)
            else:
                new_tokens.append(token)
        return new_tokens


# Shifts volume
@Transform.register("volume-shift")
class VolumeShift(Transform):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, tokens):
        k = randint(self.min, self.max)
        new_tokens = []
        for token in tokens:
            token_type, *values = token.split(':')
            if token_type == 'note':
                pitch_value, volume_value = int(values[0]), int(values[1])
                new_volume = volume_value + k
                new_volume = max(new_volume, 0)
                new_volume = min(new_volume, 127)
                new_token = 'note:%d:%d' % (pitch_value, new_volume)
                new_tokens.append(new_token)
            else:
                new_tokens.append(token)
        return new_tokens


# Shifts pitch
@Transform.register("pitch-shift")
class PitchShift(Transform):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, tokens):
        k = randint(self.min, self.max)
        new_tokens = []
        for token in tokens:
            token_type, *values = token.split(':')
            if token_type == 'note':
                pitch_value, volume_value = int(values[0]), int(values[1])
                new_pitch = pitch_value + k
                new_pitch = max(new_pitch, 0)
                new_pitch = min(new_pitch, 127)
                new_token = 'note:%d:%d' % (new_pitch, volume_value)
                new_tokens.append(new_token)
            else:
                new_tokens.append(token)
        return new_tokens
