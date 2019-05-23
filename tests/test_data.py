import torch

from squawkbox.data import TOKEN_TO_IDX, MidiDataset, pad_and_combine_instances


def test_read_instance():
    midi_dataset = MidiDataset('tests/fixtures/example.txt')

    # Check that dataset contains the correct number of instances.
    assert len(midi_dataset) == 1

    # Check that fields of the first instance are correct.
    instance = midi_dataset[0]

    # Check that src tensor is correct.
    expected_src = ['start', 'wait:1' ,'note:0:0']
    expected_src_tensor = torch.LongTensor([TOKEN_TO_IDX[x] for x in expected_src])
    assert torch.equal(instance['src'], expected_src_tensor)

    # Check that tgt tensor is correct.
    expected_tgt = ['wait:1', 'note:0:0', 'end']
    expected_tgt_tensor = torch.LongTensor([TOKEN_TO_IDX[x] for x in expected_tgt])
    assert torch.equal(instance['tgt'], expected_tgt_tensor)

    # Check that timestamp tensor is correct.
    expected_timestamp = [0, 0, 1]
    expected_timestamp_tensor = torch.FloatTensor(expected_timestamp)
    assert torch.equal(instance['timestamp'], expected_timestamp_tensor)


def test_pad_and_combine_instances():
    batch = [
        {
            'a': torch.tensor([1,2,3], dtype=torch.int64),
            'b': torch.tensor(1, dtype=torch.float32)
        },
        {
            'a': torch.tensor([1,2], dtype=torch.int64),
            'b': torch.tensor(2, dtype=torch.float32)
        }
    ]
    out_dict = pad_and_combine_instances(batch)

    # Check that output has the correct fields
    assert set(out_dict.keys()) == {'a', 'b'}

    # Check that sequences handled correctly
    expected_a = torch.tensor([[1,2,3], [1,2,0]], dtype=torch.int64)
    assert torch.equal(out_dict['a'], expected_a)

    # Check that scalars handled correctly
    expected_b = torch.tensor([1,2], dtype=torch.float32)
    assert torch.equal(out_dict['b'], expected_b)