import re


def split_long_text(text, chunk_size):
    """Split a long piece of text without cutting words."""

    words = text.split()

    parts = []
    current = ""

    for word in words:

        if not current:
            current = word

        elif len(current) + 1 + len(word) <= chunk_size:
            current += " " + word

        else:
            parts.append(current)
            current = word

    if current:
        parts.append(current)

    return parts


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Create chunks while preserving:
    - paragraphs
    - bullet points
    - complete words
    - useful overlap

    No content is intentionally discarded. Every chunk, including its
    overlap, is at most `chunk_size` characters (unless a single word is
    longer than that).
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Overlap is prepended later, so the "new" text in each chunk gets
    # whatever budget is left.
    full_size = chunk_size
    chunk_size = chunk_size - max(overlap, 0)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()

    if not text:
        return []

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Treat page markers as part of the following content
    # rather than as standalone chunks.
    text = re.sub(
        r"\n*\s*(========== PAGE \d+ ==========\s*)\n*",
        r"\n",
        text
    )

    # Split into lines first
    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    chunks = []
    current_lines = []
    current_length = 0

    for line in lines:

        # If this individual line is too large,
        # split it by words.
        if len(line) > chunk_size:

            # Save whatever we already have
            if current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_length = 0

            long_parts = split_long_text(line, chunk_size)

            for part in long_parts:
                chunks.append(part)

            continue

        # Check whether this line fits
        additional_length = (
            len(line)
            if not current_lines
            else len(line) + 1
        )

        if current_length + additional_length <= chunk_size:

            current_lines.append(line)
            current_length += additional_length

        else:

            # Save current chunk
            if current_lines:
                chunks.append("\n".join(current_lines))

            current_lines = [line]
            current_length = len(line)

    # Final chunk
    if current_lines:
        chunks.append("\n".join(current_lines))

    # --------------------------------------------------
    # Add overlap WITHOUT cutting through words/lines
    # --------------------------------------------------

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    final_chunks = [chunks[0]]

    for i in range(1, len(chunks)):

        previous = chunks[i - 1]
        current = chunks[i]

        previous_lines = previous.split("\n")

        overlap_lines = []
        overlap_length = 0

        # Take complete lines from the end
        for line in reversed(previous_lines):

            if overlap_length + len(line) + 1 > overlap:
                break

            overlap_lines.insert(0, line)
            overlap_length += len(line) + 1

        combined = "\n".join(overlap_lines) + "\n" + current if overlap_lines else current

        if len(combined) > full_size:
            combined = current

        final_chunks.append(combined)

    return final_chunks


def chunk_pages(
    pages: list[tuple[int | None, str]],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[tuple[int | None, str]]:
    """Chunk each page separately so every chunk knows its page number.

    Trade-off: no overlap across a page boundary, so a sentence split
    across two pages lands in two chunks. In exchange, citations can
    point at a real page.
    """
    chunks = []
    for page_number, text in pages:
        for chunk in chunk_text(text, chunk_size, overlap):
            chunks.append((page_number, chunk))
    return chunks

def chunk_sections(sections, chunk_size: int = 800, overlap: int = 150):
    """Chunk each section separately: every chunk keeps its page and heading.

    `sections` are document_reader.Section objects. Returns a list of
    (page, heading, text). Chunks never span two sections, so a chunk is
    never labelled with the wrong heading or page.
    """
    chunks = []
    for section in sections:
        for chunk in chunk_text(section.text, chunk_size, overlap):
            chunks.append((section.page, section.heading, chunk))
    return chunks


def contextualize(title: str, heading: str | None, text: str) -> str:
    """The text that gets embedded and keyword-indexed for a chunk.

    Prefixing the document title and section heading puts words like
    "Cybersecurity Framework" or "Password Verifiers" into every chunk under
    them, even when the chunk's own sentences never repeat those words.
    The stored/displayed text stays the original chunk.
    """
    header = title if not heading or heading == title else f"{title} > {heading}"
    return f"{header}\n{text}"
