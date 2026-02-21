import logging

from radiology_vqa.rag.document import Document, DocumentMeta
from radiology_vqa.schema import KGTriple

logger = logging.getLogger(__name__)

_DISEASE_TEMPLATES: dict[str, str] = {
    "symptom": "{entity} symptoms include: {value}",
    "cause": "{entity} is caused by: {value}",
    "treatment": "Treatment for {entity}: {value}",
    "location": "{entity} is located in the {value}",
    "description": "{entity}: {value}",
    "prevention": "Prevention of {entity}: {value}",
    "infectivity": "{entity} infectivity: {value}",
    "susceptible_population": "{entity} commonly affects: {value}",
    "alias": "{entity} is also known as: {value}",
}

_ORGAN_TEMPLATES: dict[str, str] = {
    "function": "The function of {entity}: {value}",
    "position": "{entity} is located at: {value}",
    "definition": "{entity}: {value}",
    "component": "{entity} consists of: {value}",
    "effect": "The effect of {entity}: {value}",
    "shape": "The shape of {entity}: {value}",
    "nature": "{entity} nature: {value}",
    "characteristic": "{entity} characteristic: {value}",
    "length": "The length of {entity}: {value}",
    "classification": "{entity} classification: {value}",
    "weight": "The weight of {entity}: {value}",
    "diameter": "The diameter of {entity}: {value}",
    "alias": "{entity} is also known as: {value}",
}


def _slugify(text: str) -> str:
    """Create a safe doc_id component from arbitrary text."""
    return text.lower().replace(" ", "_").replace("/", "_").replace("#", "_")


def _make_doc_id(source_type: str, entity: str, attribute: str, chunk_index: int) -> str:
    return f"{source_type}_{_slugify(entity)}_{_slugify(attribute)}_{chunk_index}"


class KGProcessor:
    """Transform SLAKE KG triples into natural language documents for indexing."""

    def process_diseases(self, triples: list[KGTriple]) -> list[Document]:
        """Create per-attribute and summary documents for each disease."""
        # Group triples by entity
        entity_attrs: dict[str, dict[str, str]] = {}
        for triple in triples:
            if triple.category != "disease":
                continue
            entity = triple.head
            attr = triple.relation
            val = triple.tail
            if entity not in entity_attrs:
                entity_attrs[entity] = {}
            entity_attrs[entity][attr] = val

        documents: list[Document] = []

        for entity, attrs in entity_attrs.items():
            # Per-attribute documents
            for attr, value in attrs.items():
                template = _DISEASE_TEMPLATES.get(attr, "{entity} {attr}: {value}")
                text = template.format(entity=entity, attr=attr, value=value)
                doc = Document(
                    text=text,
                    meta=DocumentMeta(
                        source_type="kg_disease",
                        entity_name=entity,
                        attribute=attr,
                        source_file="en_disease.csv",
                        chunk_index=0,
                    ),
                    doc_id=_make_doc_id("kg_disease", entity, attr, 0),
                )
                documents.append(doc)

            # Summary document
            summary = self._disease_summary(entity, attrs)
            if summary:
                doc = Document(
                    text=summary,
                    meta=DocumentMeta(
                        source_type="kg_disease",
                        entity_name=entity,
                        attribute="summary",
                        source_file="en_disease.csv",
                        chunk_index=0,
                    ),
                    doc_id=_make_doc_id("kg_disease", entity, "summary", 0),
                )
                documents.append(doc)

        return documents

    def process_organs(self, triples: list[KGTriple]) -> list[Document]:
        """Create per-attribute and summary documents for each organ."""
        entity_attrs: dict[str, dict[str, str]] = {}
        for triple in triples:
            if triple.category != "organ":
                continue
            entity = triple.head
            attr = triple.relation
            val = triple.tail
            if entity not in entity_attrs:
                entity_attrs[entity] = {}
            entity_attrs[entity][attr] = val

        documents: list[Document] = []

        for entity, attrs in entity_attrs.items():
            # Per-attribute documents
            for attr, value in attrs.items():
                template = _ORGAN_TEMPLATES.get(attr, "{entity} {attr}: {value}")
                text = template.format(entity=entity, attr=attr, value=value)
                doc = Document(
                    text=text,
                    meta=DocumentMeta(
                        source_type="kg_organ",
                        entity_name=entity,
                        attribute=attr,
                        source_file="en_organ.csv",
                        chunk_index=0,
                    ),
                    doc_id=_make_doc_id("kg_organ", entity, attr, 0),
                )
                documents.append(doc)

            # Summary document
            summary = self._organ_summary(entity, attrs)
            if summary:
                doc = Document(
                    text=summary,
                    meta=DocumentMeta(
                        source_type="kg_organ",
                        entity_name=entity,
                        attribute="summary",
                        source_file="en_organ.csv",
                        chunk_index=0,
                    ),
                    doc_id=_make_doc_id("kg_organ", entity, "summary", 0),
                )
                documents.append(doc)

        return documents

    def process_organ_relations(self, triples: list[KGTriple]) -> list[Document]:
        """One document per organ-system relation triple."""
        documents: list[Document] = []
        for triple in triples:
            if triple.category != "organ_rel":
                continue
            text = f"{triple.head} belongs to the {triple.tail}."
            doc = Document(
                text=text,
                meta=DocumentMeta(
                    source_type="kg_organ_rel",
                    entity_name=triple.head,
                    attribute="belong_to",
                    source_file="en_organ_rel.csv",
                    chunk_index=0,
                ),
                doc_id=_make_doc_id("kg_organ_rel", triple.head, "belong_to", 0),
            )
            documents.append(doc)
        return documents

    def process_all(self, triples: list[KGTriple]) -> list[Document]:
        """Process all triples by category and return combined document list."""
        disease_docs = self.process_diseases(triples)
        organ_docs = self.process_organs(triples)
        rel_docs = self.process_organ_relations(triples)

        logger.info(
            "KG processing: %d disease docs, %d organ docs, %d relation docs",
            len(disease_docs),
            len(organ_docs),
            len(rel_docs),
        )
        return disease_docs + organ_docs + rel_docs

    # ------------------------------------------------------------------ helpers

    def _disease_summary(self, entity: str, attrs: dict[str, str]) -> str:
        parts = [entity]
        if "description" in attrs:
            parts.append(attrs["description"])
        if "location" in attrs:
            parts.append(f"Location: {attrs['location']}")
        if "symptom" in attrs:
            parts.append(f"Symptoms: {attrs['symptom']}")
        if "cause" in attrs:
            parts.append(f"Causes: {attrs['cause']}")
        if "treatment" in attrs:
            parts.append(f"Treatment: {attrs['treatment']}")
        if "prevention" in attrs:
            parts.append(f"Prevention: {attrs['prevention']}")
        if "infectivity" in attrs:
            parts.append(f"Infectivity: {attrs['infectivity']}")
        if "susceptible_population" in attrs:
            parts.append(f"Commonly affects: {attrs['susceptible_population']}")
        return ". ".join(parts) + "."

    def _organ_summary(self, entity: str, attrs: dict[str, str]) -> str:
        parts = [entity]
        if "definition" in attrs:
            parts.append(attrs["definition"])
        if "function" in attrs:
            parts.append(f"Function: {attrs['function']}")
        if "position" in attrs:
            parts.append(f"Position: {attrs['position']}")
        if "component" in attrs:
            parts.append(f"Components: {attrs['component']}")
        if "effect" in attrs:
            parts.append(f"Effect: {attrs['effect']}")
        return ". ".join(parts) + "."
