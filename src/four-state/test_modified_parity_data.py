from modified_parity_data import (
    modified_parity_label,
    ParityStreamTrain,
    ParityStreamVal,
    BufferedShuffle
)


def test_modified_parity_label_basic():
    assert modified_parity_label([0]) == 0
    assert modified_parity_label([1]) == 1
    assert modified_parity_label([0, 2]) == 0
    assert modified_parity_label([1, 2, 2, 0, 0, 1]) == 1
    assert modified_parity_label([0, 2, 2]) == 1
    assert modified_parity_label([1, 2]) == 1
    assert modified_parity_label([0, 2, 2]) == 1


def test_train_first_token_not_two():
    ds = ParityStreamTrain(L_train=4)
    for seq, _ in ds:
        assert seq[0].item() in (0, 1), "Train sequence started with 2!"


def test_train_label_matching():
    ds = ParityStreamTrain(L_train=5)
    for seq, lbl in ds:
        assert lbl.item() == modified_parity_label(seq.tolist())


def test_val_length_and_first_token():
    ds = ParityStreamVal(num_val=100, L_val=8, seed=123)
    for seq, _ in ds:
        assert len(seq) == 8
        assert seq[0].item() in (0, 1), "Val sequence started with 2!"


def test_val_label_matching():
    ds = ParityStreamVal(num_val=100, L_val=8, seed=456)
    for seq, lbl in ds:
        assert lbl.item() == modified_parity_label(seq.tolist())


def test_buffered_shuffle_preserves_data():
    base = list(ParityStreamTrain(L_train=3))
    shuffled = list(BufferedShuffle(
        ParityStreamTrain(L_train=3), buffer_size=50))
    base_set = {tuple(s.tolist()) for s, _ in base}
    shuf_set = {tuple(s.tolist()) for s, _ in shuffled}
    assert base_set == shuf_set
    assert len(base) == len(shuffled)


if __name__ == "__main__":
    test_modified_parity_label_basic()
    test_train_first_token_not_two()
    test_train_label_matching()
    test_val_length_and_first_token()
    test_val_label_matching()
    test_buffered_shuffle_preserves_data()
    print("All tests passed!")
