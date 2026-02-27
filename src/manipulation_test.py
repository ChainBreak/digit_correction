import pytest

from src.manipulation import manipulate, get_opposite_command


@pytest.mark.parametrize(
    "s,command,expected",
    [
        # edit
        ("hello", "E1,a", "hallo"),
        ("hello", "E0,x", "xello"),
        ("hello", "E4,!", "hell!"),
        ("abc", "E1,b", "abc"),
        ("123456", "E1,5", "153456"),
        # delete
        ("hello", "D0", "ello"),
        ("hello", "D1", "hllo"),
        ("hello", "D4", "hell"),
        ("ab", "D0", "b"),
        ("123456", "D2", "12456"),
        # insert
        ("hello", "I0,x", "xhello"),
        ("hello", "I1,-", "h-ello"),
        ("hello", "I5,!", "hello!"),
        ("ab", "I1,x", "axb"),
        ("123456", "I2,3", "1233456"),
        # unknown/invalid command returns unchanged
        ("hello", "F", "hello"),
        ("hello", "X", "hello"),
        ("hello", "E", "hello"),
    ],
)
def test_manipulate(s, command, expected):
    assert manipulate(s, command) == expected

@pytest.mark.parametrize(
    "s,command",
    [
        ("hello", "E1,a"),
        ("hello", "E4,x"),
        ("hello", "D0"),
        ("hello", "D4"),
        ("hello", "I0,x"),
        ("hello", "I5,!"),
        ("abc", "E1,b"),
        ("abc", "D1"),
        ("abc", "I1,x"),
        ("abc", "I2,y"),
        ("123456", "E1,5"),
        ("123456", "D2"),
        ("123456", "I2,3"),
    ],
)
def test_manipulate_round_trip(s, command):

    opposite_command = get_opposite_command(s, command)

    manipulated_s = manipulate(s, command)

    assert manipulate(manipulated_s, opposite_command) == s