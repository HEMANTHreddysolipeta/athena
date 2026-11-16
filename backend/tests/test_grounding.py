from athena.grounding import check_grounding, specific_tokens, split_sentences

SOURCES = [
    "Verifiers SHALL require passwords to be a minimum of 15 characters. At AAL3 the timeout is 12 hours.",
    "ClickSlice has a page authority of 31 and a domain authority of 25.",
]


def test_correct_answer_is_supported():
    g = check_grounding("Passwords need at least 15 characters [1]. The AAL3 timeout is 12 hours [1].", SOURCES)
    assert g.supported and g.unsupported == [] and g.checked == 3


def test_invented_numbers_are_flagged():
    """The Milestone 1 failure: the model wrote 28/45, the page says 31/25."""
    g = check_grounding("ClickSlice has page authority 28 and domain authority 45 [2].", SOURCES)
    assert not g.supported
    assert g.unsupported == ["28", "45"]


def test_number_words_match_digits():
    assert check_grounding("Passwords need fifteen characters [1].", SOURCES).supported


def test_citing_the_wrong_source_is_not_an_invented_fact():
    g = check_grounding("The minimum is 15 characters [2].", SOURCES)
    assert g.supported


def test_citations_to_missing_sources_are_flagged():
    g = check_grounding("See source [7].", SOURCES)
    assert not g.supported and g.invalid_citations == [7]


def test_uncited_claims_are_counted_and_checked_against_everything():
    g = check_grounding("The timeout at AAL3 is 12 hours.", SOURCES)
    assert g.supported and g.uncited_sentences == 1


def test_invented_acronym_is_flagged():
    g = check_grounding("This is required at AAL4 [1].", SOURCES)
    assert g.unsupported == ["AAL4"]


def test_plain_prose_has_nothing_to_check():
    g = check_grounding("Passwords should be long and memorable [1].", SOURCES)
    assert g.supported and g.checked == 0


def test_specific_tokens():
    assert specific_tokens("NIST SP 800-63B needs fifteen characters at AAL2 [1].") == [
        "800-63B", "15", "AAL2", "NIST", "SP"]
    assert specific_tokens("It costs 1,000 dollars.") == ["1,000"]


def test_sentence_split_keeps_list_items_apart():
    assert split_sentences("First [1].\n- item two [2]\nThird. Fourth.") == [
        "First [1].", "- item two [2]", "Third.", "Fourth."]


def test_single_digits_are_checked_but_list_markers_are_not():
    assert specific_tokens("1. The framework has 6 functions [1].") == ["6"]
    assert check_grounding("2) It has 7 pillars [1].", SOURCES).unsupported == ["7"]
