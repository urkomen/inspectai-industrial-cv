"""Taxonomia unificada de clases (configs/taxonomy.yaml)."""
from dataclasses import dataclass
from functools import lru_cache

import yaml

from src.utils.paths import CONFIGS


@dataclass(frozen=True)
class ClassDef:
    id: int
    name: str
    dataset: str
    source_label: str
    family: str


class Taxonomy:
    def __init__(self, classes, families):
        self.classes = classes
        self.families = families
        self._by_source = {(c.dataset, c.source_label): c for c in classes}

    @property
    def names(self):
        """Nombres de clase ordenados por id (el orden que espera YOLO)."""
        return [c.name for c in self.classes]

    def lookup(self, dataset, source_label):
        """ClassDef de una etiqueta de origen, o None si no esta en la taxonomia."""
        return self._by_source.get((dataset, source_label))

    def for_dataset(self, dataset):
        return [c for c in self.classes if c.dataset == dataset]


def _validate(classes, families):
    if [c.id for c in classes] != list(range(len(classes))):
        raise ValueError("taxonomy.yaml: los ids deben ser 0..N-1 sin huecos y en orden")
    if len({c.name for c in classes}) != len(classes):
        raise ValueError("taxonomy.yaml: nombres de clase duplicados")
    if len({(c.dataset, c.source_label) for c in classes}) != len(classes):
        raise ValueError("taxonomy.yaml: etiqueta de origen duplicada dentro de un dataset")
    bad = {c.family for c in classes} - set(families)
    if bad:
        raise ValueError(f"taxonomy.yaml: familias no declaradas: {sorted(bad)}")


@lru_cache(maxsize=1)
def load_taxonomy():
    cfg = yaml.safe_load((CONFIGS / "taxonomy.yaml").read_text(encoding="utf-8"))
    classes = [ClassDef(**c) for c in cfg["classes"]]
    _validate(classes, cfg["families"])
    return Taxonomy(classes, cfg["families"])
