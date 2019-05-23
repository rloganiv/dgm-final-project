Squawkbox
===



Transformer models for MIDI generation in PyTorch.
Created as the final project for Deep Generative Models, Spring 2019.


Getting Started
---
Begin by installing the requirements:
```{bash}
pip install -r requirements.txt
```

Next install `squawkbox` by running:
```{bash}
pip install -e .
```


Data
---
Models are trained/evaluated on the MAESTRO v2.0.0 dataset:

https://magenta.tensorflow.org/datasets/maestro#v200

To parse the MIDI to sequences of tokens run:

```{bash}
python squawkbox/commands/process_midi.py process-maestro \
    [PATH TO maestro-v2.0.0.csv] \
    [OUTPUT DIR]
```
this will create three files: `train.txt`, `validation.txt` and `test.txt` using the official MAESTRO splits. Each line contains a single song from the dataset represented as a sequence of `wait` and `note` events.