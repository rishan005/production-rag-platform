from dataclasses import dataclass, field
import os
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

@dataclass
class Page:
  """One page's worth of extrCTED CONTENT - the atomic unit before chunking."""
  page_number: int
  text: str
  char_count: int = 0

  def __post_init__(self):
    self.char_count = len(self.text)

@dataclass
class ParsedDocument:
  """The common internal representative evry file type gets normalized into."""
  document_id: str
  filename: str
  source: str
  pages: list[Page] = field(default_factory=list)
  doc_type: str = "pdf"
  metadata: dict = field(default_factory=dict)

  @property
  def full_text(self) -> str:
    return "\n\n".join(p.text for p in self.pages)

  @property
  def needs_ocr_pages(self) -> list[int]:
    """Pages with suspiciously little text - flagged for OCR in Stage 2."""
    return [p.page_number for p in self.pages if p.char_count <20]


def _make_document_id(filename: str) -> str:
  """Deterministic short ID so re-running the notebook on the same file is stable."""
  return hashlib.md5(filename.encode()).hexdigest()[:8]


def parse_pdf(path: str) -> ParsedDocument:
  """
  Extracts text per-page from a PDF using PyMuPDF.
  Does Not do OCR yet - that's Stage 2. Pages with little/no extractable
  text (e.g. scanned images) are just flagged via 'needs_ocr_pages'.
  """
  filename = os.path.basename(path)
  doc_id = _make_document_id(filename)

  pdf = pymupdf.open(path)
  pages = []
  for i, page in enumerate(pdf, start=1):
    text = page.get_text("text")
    pages.append(Page(page_number=i, text=text.strip()))
  pdf.close()

  return ParsedDocument(
      document_id=doc_id,
      filename=filename,
      source=path,
      pages=pages,
      doc_type="pdf",
      metadata={"num_pages":len(pages)},
  )

def parse_documents(paths:list[str]) ->list[ParsedDocument]:
  """Dispatches to the right parser per file extention.OnluPDF wired up in Stage 1."""
  parsed = []
  for path in paths:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
      parsed.append(parse_pdf(path))
    else:
      print(f"skipping {path}: parser for '{ext}' arrives in a later stage (DOCX?HTML?TXT.")
  return parsed

##parsed_docs = parse_documents(document_paths)

def _find_repeated_lines(pages: list[Page], min_page_fraction: float = 0.6) -> set[str]:
    """
    Detects lines that repeat across most pages - these are almost always
    running headers/footers (e.g. "Employee Handbook - Confidential", "Page 3 of 20")
    rather than real content, so we flag them for removal.
    """
    counter = Counter()
    for p in pages:
        lines = [l.strip() for l in p.text.split("\n") if l.strip()]
        edge_lines = lines[:2] + lines[-2:]  # headers/footers live at page edges
        counter.update(set(edge_lines))

    threshold = max(2, int(len(pages) * min_page_fraction))
    return {line for line, count in counter.items() if count >= threshold and len(line) < 100}


def clean_page_text(text: str, repeated_lines: set[str]) -> str:
    """Cleans one page's text while preserving structural cues (headings, numbering)."""
    lines = [line for line in text.split("\n") if line.strip() not in repeated_lines]
    text = "\n".join(lines)

    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)      # dehyphenate wrapped words
    text = re.sub(r"\n{3,}", "\n\n", text)                # collapse excess blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)                 # collapse repeated spaces/tabs
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


def clean_document(doc: ParsedDocument) -> ParsedDocument:
    """Cleans every page of a document, using cross-page repeated-line detection."""
    repeated = _find_repeated_lines(doc.pages)
    for p in doc.pages:
        p.text = clean_page_text(p.text, repeated)
        p.char_count = len(p.text)
    doc.metadata["repeated_lines_removed"] = list(repeated)
    return doc
