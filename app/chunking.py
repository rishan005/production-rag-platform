import re
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion import Page, ParsedDocument
from app.config import Config

HEADING_PATTERNS = [
    re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z].{0,80}$"),        
    re.compile(r"^[A-Z][A-Z\s&/\-]{3,60}$"),                 
    re.compile(r"^(Section|Chapter|Article)\s+\d+.*$", re.IGNORECASE),
]

def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90:
        return False
    return any(pattern.match(line) for pattern in HEADING_PATTERNS)


def extract_sections(page: Page) -> list[tuple[int, str]]:
    """Returns (character_offset, heading_text) for every detected heading on this page."""
    sections, offset = [], 0
    for line in page.text.split("\n"):
        if is_heading(line):
            sections.append((offset, line.strip()))
        offset += len(line) + 1
    return sections


def section_at_offset(sections: list[tuple[int, str]], offset: int, fallback: str = "General") -> str:
    """Finds the most recent heading at or before a given character offset."""
    current = fallback
    for sect_offset, sect_text in sections:
        if sect_offset <= offset:
            current = sect_text
        else:
            break
    return current

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    source: str
    page: int
    section: str
    text: str

    # Metadata for filtering and permissions
    document_type: str = "general"
    access_level: str = "public"


def _make_chunk_id(document_id: str, page: int, index: int) -> str:
    return f"{document_id}_p{page}_c{index}"


def chunk_document(
    doc: ParsedDocument,
    config: Config,
    document_type: str = "general",
    access_level: str = "public",
) -> list[Chunk]:
    """
    Splits every page's cleaned text into overlapping chunks, tagging each
    chunk with its page number and nearest preceding section heading.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in doc.pages:
        if not page.text.strip():
            continue  

        sections = extract_sections(page)
        page_chunks = splitter.split_text(page.text)

        cursor = 0
        for i, chunk_text in enumerate(page_chunks):
            offset = page.text.find(chunk_text, cursor)
            if offset == -1:
                offset = cursor
            section = section_at_offset(sections, offset)

            chunks.append(Chunk(
                chunk_id=_make_chunk_id(
                doc.document_id,
                page.page_number,
                i
            ),
                document_id=doc.document_id,
                filename=doc.filename,
                source=doc.source,
                page=page.page_number,
                section=section,
                text=chunk_text,

                document_type=document_type,
                access_level=access_level,
            ))
            cursor = offset + len(chunk_text)

    return chunks