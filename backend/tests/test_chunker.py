import pytest

from athena.chunker import chunk_pages, chunk_text

LOREM = (
    "Retrieval augmented generation combines search with a language model. "
    "The retriever finds relevant passages and the generator writes an answer. "
)


def words(text: str) -> list[str]:
    return text.split()


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", 100, 20) == ["hello world"]


@pytest.mark.parametrize("size,overlap", [(200, 40), (500, 100), (800, 150)])
def test_chunks_never_exceed_size_including_overlap(size, overlap):
    text = "\n".join(LOREM * (i % 3 + 1) for i in range(40))
    chunks = chunk_text(text, size, overlap)
    assert len(chunks) > 1
    assert max(len(c) for c in chunks) <= size


def test_no_words_are_lost_or_cut():
    lines = [f"line {i} " + LOREM for i in range(30)]
    text = "\n".join(lines)
    chunks = chunk_text(text, 300, 0)  # no overlap: chunks should re-join exactly
    assert words(" ".join(chunks)) == words(text)


def test_overlap_repeats_end_of_previous_chunk():
    lines = [f"sentence number {i} about retrieval." for i in range(40)]
    chunks = chunk_text("\n".join(lines), 200, 60)
    last_line_of_first = chunks[0].split("\n")[-1]
    assert chunks[1].startswith(last_line_of_first)


def test_very_long_line_is_split_on_word_boundaries():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, 200, 0)
    assert all(len(c) <= 200 for c in chunks)
    assert words(" ".join(chunks)) == words(text)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", 100, 100)


def test_chunk_pages_keeps_page_numbers():
    pages = [(1, LOREM * 6), (2, "short page two")]
    chunks = chunk_pages(pages, 200, 40)
    assert {p for p, _ in chunks} == {1, 2}
    assert chunks[-1] == (2, "short page two")
