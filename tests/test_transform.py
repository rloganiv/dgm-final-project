from squawkbox.transform import TimeStretch, PitchShift, VolumeShift



def test_transform():
    example = "start note:71:60 wait:93 note:71:0 wait:86 end"
    tokens = example.strip().split()
    transformer = TimeStretch(1.5, 1.5)
    transformed_tokens = transformer(tokens)
    assert transformed_tokens[0] == "start"
    assert transformed_tokens[1] == "note:71:60"
    assert transformed_tokens[2] == "wait:140"
    assert transformed_tokens[3] == "note:71:0"
    assert transformed_tokens[4] == "wait:129"
    assert transformed_tokens[5] == "end"

    transformer = TimeStretch(100, 100)
    transformed_tokens = transformer(tokens)
    assert transformed_tokens[0] == "start"
    assert transformed_tokens[1] == "note:71:60"
    assert transformed_tokens[2] == "wait:4095"
    assert transformed_tokens[3] == "note:71:0"
    assert transformed_tokens[4] == "wait:4095"
    assert transformed_tokens[5] == "end"

    transformer = PitchShift(10,10)
    transformed_tokens = transformer(tokens)
    assert transformed_tokens[0] == "start"
    assert transformed_tokens[1] == "note:81:60"
    assert transformed_tokens[2] == "wait:93"
    assert transformed_tokens[3] == "note:81:0"
    assert transformed_tokens[4] == "wait:86"
    assert transformed_tokens[5] == "end"

    transformer = PitchShift(80,80)
    transformed_tokens = transformer(tokens)
    assert transformed_tokens[0] == "start"
    assert transformed_tokens[1] == "note:127:60"
    assert transformed_tokens[2] == "wait:93"
    assert transformed_tokens[3] == "note:127:0"
    assert transformed_tokens[4] == "wait:86"
    assert transformed_tokens[5] == "end"

    transformer = PitchShift(-80,-80)
    transformed_tokens = transformer(tokens)
    assert transformed_tokens[0] == "start"
    assert transformed_tokens[1] == "note:0:60"
    assert transformed_tokens[2] == "wait:93"
    assert transformed_tokens[3] == "note:0:0"
    assert transformed_tokens[4] == "wait:86"
    assert transformed_tokens[5] == "end"

    transformer = VolumeShift(5,5)
    transformed_tokens = transformer(tokens)
    assert transformed_tokens[0] == "start"
    assert transformed_tokens[1] == "note:71:65"
    assert transformed_tokens[2] == "wait:93"
    assert transformed_tokens[3] == "note:71:5"
    assert transformed_tokens[4] == "wait:86"
    assert transformed_tokens[5] == "end"


    transformer = VolumeShift(80,80)
    transformed_tokens = transformer(tokens)
    assert transformed_tokens[0] == "start"
    assert transformed_tokens[1] == "note:71:127"
    assert transformed_tokens[2] == "wait:93"
    assert transformed_tokens[3] == "note:71:80"
    assert transformed_tokens[4] == "wait:86"
    assert transformed_tokens[5] == "end"

    transformer = VolumeShift(-80,-80)
    transformed_tokens = transformer(tokens)
    assert transformed_tokens[0] == "start"
    assert transformed_tokens[1] == "note:71:0"
    assert transformed_tokens[2] == "wait:93"
    assert transformed_tokens[3] == "note:71:0"
    assert transformed_tokens[4] == "wait:86"
    assert transformed_tokens[5] == "end"
