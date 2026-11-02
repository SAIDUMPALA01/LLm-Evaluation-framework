import pytest

from sixdim_eval.parsing import JudgeOutputError, extract_json_object


def test_plain_json():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_markdown_fence():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_prose_around_json():
    assert extract_json_object('Here you go:\n{"a": {"b": 2}}\nHope that helps!') == {"a": {"b": 2}}


@pytest.mark.parametrize("raw", ["", "no json here", "[1, 2, 3]", "{broken"])
def test_rejects_bad_output(raw):
    with pytest.raises(JudgeOutputError):
        extract_json_object(raw)
