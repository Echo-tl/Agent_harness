"""robust_json_parse 容错解析测试。"""

from utils.parsing import robust_json_parse


def test_plain_json_object():
    assert robust_json_parse('{"a": 1, "b": 2}') == {"a": 1, "b": 2}


def test_markdown_codeblock():
    text = '```json\n{"keep": true, "score": 8}\n```'
    assert robust_json_parse(text) == {"keep": True, "score": 8}


def test_trailing_comma_repaired():
    text = '{"keep": true, "score": 8,}'
    assert robust_json_parse(text) == {"keep": True, "score": 8}


def test_json_array():
    assert robust_json_parse('[1, 2, 3]') == [1, 2, 3]


def test_invalid_returns_none():
    assert robust_json_parse("这不是 JSON") is None
    assert robust_json_parse("") is None
    assert robust_json_parse(None) is None
