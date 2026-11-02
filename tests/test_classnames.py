"""Class-name swap (DECISIONS.md D-25): the two names are exchanged on reading, nothing else changes."""
from qpi.utils.classnames import (CLASSES, LEGACY_NAMES, MANUSCRIPT_NAMES, legacy_name, swap_obj, swap_text,
                                  swap_text_report)


def test_orders():
    assert LEGACY_NAMES[2:4] == ["echinocyte", "spherocyte"] and MANUSCRIPT_NAMES[2:4] == ["spherocyte", "echinocyte"]
    assert LEGACY_NAMES[0] == MANUSCRIPT_NAMES[0] and LEGACY_NAMES[1] == MANUSCRIPT_NAMES[1] and LEGACY_NAMES[4] == MANUSCRIPT_NAMES[4]
    assert CLASSES == MANUSCRIPT_NAMES[1:]
    assert [swap_text(n) for n in LEGACY_NAMES] == MANUSCRIPT_NAMES


def test_swap_is_an_involution_and_keeps_other_text():
    keys = {"dice_echinocyte": 0.1, "ipe_spherocyte_pct": 2.0, "dice_discocyte": 0.9, "x": ["echinocyte", {"Spherocyte": 1}]}
    once = swap_obj(keys)
    assert once == {"dice_spherocyte": 0.1, "ipe_echinocyte_pct": 2.0, "dice_discocyte": 0.9, "x": ["spherocyte", {"Echinocyte": 1}]}
    assert swap_obj(once) == keys
    assert swap_text("stomatocyte, discocyte") == "stomatocyte, discocyte"


def test_report_abbreviation_and_legacy_name():
    assert swap_text_report("| Dice echi | Dice stom |") == "| Dice sphero | Dice stom |"
    assert legacy_name("spherocyte") == "echinocyte" and legacy_name("discocyte") == "discocyte"
