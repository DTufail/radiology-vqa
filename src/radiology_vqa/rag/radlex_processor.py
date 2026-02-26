"""RadLex ontology processor: converts Tier 1 RadLex entries to Document objects.

Tier 1 = entries with at least one definition (col 28 or col 3 non-empty),
a non-empty Preferred Label, and not marked obsolete.

Reads directly from the .xls file using xlrd — no LibreOffice conversion needed.

Column layout (0-based, confirmed by inspection of Radlex.xls):
  col 0:  Class ID
  col 1:  Preferred Label
  col 2:  Synonyms
  col 3:  Definitions (short)
  col 4:  Obsolete flag ('1' = obsolete)
  col 28: Definition (full; preferred over col 3)
  col 46: prefixIRI (RID identifier, e.g. 'RID4265')
"""

import logging
import re
from pathlib import Path

from radiology_vqa.rag.document import Document, DocumentMeta

logger = logging.getLogger(__name__)

_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")


def _slugify(text: str) -> str:
    """Create a safe doc_id component from arbitrary text."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


class RadLexProcessor:
    """Load RadLex ontology from XLS and produce Tier 1 Document objects.

    Tier 1 entries are those with at least one clinical definition.
    Of the 46,657 RadLex entries, approximately 3,737 are Tier 1.
    The remaining entries are label-only or synonym-only, providing
    insufficient retrieval value to justify inclusion in the index.

    Args:
        xls_path: Path to Radlex.xls (the RSNA RadLex ontology file).
    """

    def __init__(self, xls_path: str) -> None:
        self._path = xls_path

    def process(self) -> list[Document]:
        """Return Tier 1 RadLex entries as Document objects.

        Returns:
            List of Document objects, one per Tier 1 RadLex entry.
            All documents have source_type='radlex' and doc_ids prefixed
            with 'radlex_'.
        """
        try:
            import xlrd
        except ImportError as exc:
            raise ImportError(
                "xlrd is required for RadLexProcessor. Install with: pip install xlrd"
            ) from exc

        wb = xlrd.open_workbook(self._path)
        sh = wb.sheet_by_index(0)

        documents: list[Document] = []
        replacement_count = 0
        skipped_no_label = 0
        skipped_obsolete = 0
        skipped_no_def = 0

        for row_idx in range(1, sh.nrows):  # row 0 is header
            label = str(sh.cell_value(row_idx, 1)).strip()
            synonyms = str(sh.cell_value(row_idx, 2)).strip()
            short_def = str(sh.cell_value(row_idx, 3)).strip()
            obsolete = str(sh.cell_value(row_idx, 4)).strip()
            full_def = str(sh.cell_value(row_idx, 28)).strip()
            rid = str(sh.cell_value(row_idx, 46)).strip()

            # Must have a label
            if not label:
                skipped_no_label += 1
                continue

            # Must not be marked obsolete
            if obsolete == "1":
                skipped_obsolete += 1
                continue

            # Tier 1 filter: must have at least one definition
            definition = full_def or short_def
            if not definition:
                skipped_no_def += 1
                continue

            # Build document text
            text = f"{label}: {definition}"
            if synonyms:
                text += f" Also known as: {synonyms.replace('|', ', ')}"

            # Strip non-ASCII characters (occasional encoding artifacts)
            cleaned, n = _NON_ASCII_RE.subn("", text)
            replacement_count += n
            cleaned = cleaned.strip()
            if not cleaned:
                continue

            # doc_id: prefer the RID from prefixIRI column, else slugify label
            doc_id_suffix = rid if rid else _slugify(label)
            doc_id = f"radlex_{doc_id_suffix}"

            documents.append(
                Document(
                    text=cleaned,
                    meta=DocumentMeta(
                        source_type="radlex",
                        entity_name=label,
                        attribute="definition",
                        source_file=Path(self._path).name,
                        chunk_index=0,
                    ),
                    doc_id=doc_id,
                )
            )

        if replacement_count > 0:
            logger.warning(
                "RadLexProcessor: stripped %d non-ASCII characters across documents",
                replacement_count,
            )

        logger.info(
            "RadLexProcessor: %d Tier 1 documents "
            "(skipped: %d no-label, %d obsolete, %d no-definition)",
            len(documents),
            skipped_no_label,
            skipped_obsolete,
            skipped_no_def,
        )
        return documents
