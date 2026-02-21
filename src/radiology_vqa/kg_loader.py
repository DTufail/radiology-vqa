import csv
import logging
from pathlib import Path

from radiology_vqa.schema import KGTriple

logger = logging.getLogger(__name__)

_KG_FILES: dict[str, str] = {
    "en_disease.csv": "disease",
    "en_organ.csv": "organ",
    "en_organ_rel.csv": "organ_rel",
}


def load_kg(slake_dir: Path) -> list[KGTriple]:
    kg_dir = slake_dir / "KG"
    if not kg_dir.exists():
        logger.error("KG directory not found: %s", kg_dir)
        return []

    all_triples: list[KGTriple] = []
    for filename, category in _KG_FILES.items():
        file_path = kg_dir / filename
        if not file_path.exists():
            logger.warning("KG file not found: %s", file_path)
            continue
        triples = _load_kg_file(file_path, category)
        logger.info("KG %s: %d triples", filename, len(triples))
        all_triples.extend(triples)

    return all_triples


def _load_kg_file(path: Path, category: str) -> list[KGTriple]:
    # Ignore macOS resource fork files
    if path.name.startswith("._"):
        return []

    triples: list[KGTriple] = []
    try:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="#")
            header = next(reader, None)
            if header is None:
                return []

            header_lower = [h.strip().lower() for h in header]

            def find_col(names: list[str]) -> int:
                for name in names:
                    if name in header_lower:
                        return header_lower.index(name)
                return -1

            head_idx = find_col(["head", "subject", "entity1", "h"])
            rel_idx = find_col(["relation", "rel", "predicate", "r"])
            tail_idx = find_col(["tail", "object", "entity2", "t"])

            # Positional fallback
            if head_idx == -1:
                head_idx = 0
            if rel_idx == -1:
                rel_idx = 1
            if tail_idx == -1:
                tail_idx = 2

            for row in reader:
                if len(row) < 3:
                    continue
                head = row[head_idx].strip()
                rel = row[rel_idx].strip()
                tail = row[tail_idx].strip()
                if not head or not rel or not tail:
                    continue
                triples.append(KGTriple(head=head, relation=rel, tail=tail, category=category))
    except Exception as e:
        logger.error("Failed to load KG file %s: %s", path, e)

    return triples
