import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from interaction.prompt_parser import parse_interactive_blocks

def test_select():
    text = "正文\n<<SELECT: 問題>>\n- 選A\n- 選B\n<</SELECT>>"
    clean, blocks = parse_interactive_blocks(text)
    assert clean == "正文"
    assert blocks[0].type == "select"
    assert blocks[0].choices == ["選A", "選B"]

def test_no_blocks():
    clean, blocks = parse_interactive_blocks("普通回應")
    assert clean == "普通回應" and blocks == []

def test_confirm():
    text = "好的，確認一下。\n<<CONFIRM: 這是你的核心命題嗎？>><</CONFIRM>>"
    clean, blocks = parse_interactive_blocks(text)
    assert "確認一下" in clean
    assert blocks[0].type == "confirm"
    assert blocks[0].prompt == "這是你的核心命題嗎？"

def test_multi():
    text = "<<MULTI: 選擇相關概念>>\n- 規訓\n- 權力\n- 凝視\n<</MULTI>>"
    clean, blocks = parse_interactive_blocks(text)
    assert clean == ""
    assert blocks[0].type == "multi"
    assert len(blocks[0].choices) == 3

def test_input():
    text = "你提到了背面。<<INPUT: 用一句話說明你的核心想法>><</INPUT>>"
    clean, blocks = parse_interactive_blocks(text)
    assert "你提到了背面" in clean
    assert blocks[0].type == "input"

if __name__ == "__main__":
    test_select()
    test_no_blocks()
    test_confirm()
    test_multi()
    test_input()
    print("All parser tests passed!")
