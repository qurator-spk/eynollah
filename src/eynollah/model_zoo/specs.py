from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class EynollahModelSpec():
    """
    Describing a single model abstractly.
    """
    category: str
    # Relative filename to the models_eynollah directory in the dists
    filename: str
    # URL to the smallest model distribution containing this model (link to Zenodo)
    dist_url: str
    type: str
    variant: str = ''
    help: str = ''

class EynollahModelSpecSet():
    """
    List of all used models for eynollah.
    """
    specs: List[EynollahModelSpec]

    def __init__(self, specs: List[EynollahModelSpec]) -> None:
        self.specs = sorted(specs, key=lambda x: x.category + '0' + x.variant)
        self.categories: Set[str] = set([spec.category for spec in self.specs])
        self.variants: Dict[str, Set[str]] = {
            spec.category: set([x.variant for x in self.specs if x.category == spec.category])
            for spec in self.specs
        }
        self._index_category_variant: Dict[Tuple[str, str], EynollahModelSpec] = {
            (spec.category, spec.variant): spec
            for spec in self.specs
        }

    def asdict(self) -> Dict[str, Dict[str, str]]:
        return {
            spec.category: {
                spec.variant: spec.filename
            }
            for spec in self.specs
        }

    def get(self, category: str, variant: str) -> EynollahModelSpec:
        if category not in self.categories:
            raise ValueError(f"Unknown category '{category}', must be one of {self.categories}")
        if variant not in self.variants[category]:
            raise ValueError(f"Unknown variant {variant} for {category}. Known variants: {self.variants[category]}")
        return self._index_category_variant[(category, variant)]


