import re
from typing import List, Dict


class CitationTool:
    """Extracts [1], [2]… indices from answer and maps to sources."""

    _pattern = re.compile(r"\[(\d+)\]")

    def extract_indices(self, answer: str) -> List[int]:
        return sorted({int(m.group(1)) for m in self._pattern.finditer(answer)})

    def attach_sources(self, answer: str, sources: List[Dict]) -> List[Dict]:
        used = self.extract_indices(answer)
        mapped: List[Dict] = []
        for idx in used:
            if 1 <= idx <= len(sources):
                mapped.append(sources[idx - 1])
        return mapped
