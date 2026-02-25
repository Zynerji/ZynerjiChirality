"""COD crystal structure download and chiral filtering.

Queries the Crystallography Open Database REST API for structures in
the 65 Sohncke (chiral) space groups and downloads CIF files.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

# The 65 Sohncke (chiral) space groups — only these can host chiral crystals
SOHNCKE_GROUPS: set[int] = {
    1,
    3, 4, 5,
    16, 17, 18, 19, 20, 21, 22, 23, 24,
    75, 76, 77, 78, 79, 80,
    89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
    143, 144, 145, 146,
    149, 150, 151, 152, 153, 154, 155,
    168, 169, 170, 171, 172, 173,
    177, 178, 179, 180, 181, 182,
    195, 196, 197, 198, 199,
    207, 208, 209, 210, 211, 212, 213, 214,
}


@dataclass
class CrystalStructure:
    """A crystal structure entry from COD."""

    cod_id: str
    formula: str
    space_group: str
    sg_number: int
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    cif_url: str
    is_chiral_sg: bool = False

    def __post_init__(self):
        self.is_chiral_sg = self.sg_number in SOHNCKE_GROUPS

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CrystalStructure:
        return cls(**{k: v for k, v in d.items() if k != "is_chiral_sg"})


class CODDownloader:
    """Download and iterate crystal structures from the COD REST API.

    Parameters
    ----------
    delay : float
        Seconds between API requests (rate limiting).
    """

    BASE_URL = "https://www.crystallography.net/cod"

    def __init__(self, delay: float = 0.5):
        self.delay = delay
        self._last_request = 0.0

    def _rate_limit(self):
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.monotonic()

    def _fetch_json(self, url: str) -> list[dict]:
        """Fetch JSON from COD API with rate limiting."""
        self._rate_limit()
        req = Request(url, headers={"User-Agent": "ZynerjiChirality/0.5.0"})
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, json.JSONDecodeError) as e:
            logger.warning("COD API error for %s: %s", url, e)
            return []

    def search_by_space_group(
        self, sg_number: int, limit: int = 500
    ) -> list[CrystalStructure]:
        """Query COD for structures in a specific space group.

        Parameters
        ----------
        sg_number : int
            Space group number (1-230).
        limit : int
            Maximum results to return.
        """
        url = (
            f"{self.BASE_URL}/result?format=json"
            f"&sg={sg_number}"
            f"&max={limit}"
        )
        results = self._fetch_json(url)
        structures = []
        for entry in results:
            try:
                cod_id = str(entry.get("file", ""))
                if not cod_id:
                    continue
                structures.append(CrystalStructure(
                    cod_id=cod_id,
                    formula=entry.get("formula", ""),
                    space_group=entry.get("sg", ""),
                    sg_number=sg_number,
                    a=float(entry.get("a", 0)),
                    b=float(entry.get("b", 0)),
                    c=float(entry.get("c", 0)),
                    alpha=float(entry.get("alpha", 90)),
                    beta=float(entry.get("beta", 90)),
                    gamma=float(entry.get("gamma", 90)),
                    cif_url=f"{self.BASE_URL}/{cod_id}.cif",
                ))
            except (ValueError, TypeError) as e:
                logger.debug("Skipping entry: %s", e)
        return structures

    def search_chiral_structures(
        self, limit_per_sg: int = 500
    ) -> Iterator[CrystalStructure]:
        """Iterate chiral structures from all 65 Sohncke space groups.

        Parameters
        ----------
        limit_per_sg : int
            Max structures per space group.
        """
        total = 0
        for sg in sorted(SOHNCKE_GROUPS):
            logger.info("Querying COD space group %d ...", sg)
            structures = self.search_by_space_group(sg, limit=limit_per_sg)
            logger.info("  SG %d: %d structures", sg, len(structures))
            for s in structures:
                yield s
                total += 1
        logger.info("Total chiral structures discovered: %d", total)

    def search_by_formula(self, formula: str) -> list[CrystalStructure]:
        """Search COD by chemical formula."""
        url = f"{self.BASE_URL}/result?format=json&formula={formula}"
        results = self._fetch_json(url)
        structures = []
        for entry in results:
            try:
                cod_id = str(entry.get("file", ""))
                if not cod_id:
                    continue
                sg_num = int(entry.get("sgNumber", entry.get("sg", 0)))
                structures.append(CrystalStructure(
                    cod_id=cod_id,
                    formula=entry.get("formula", formula),
                    space_group=entry.get("sg", ""),
                    sg_number=sg_num,
                    a=float(entry.get("a", 0)),
                    b=float(entry.get("b", 0)),
                    c=float(entry.get("c", 0)),
                    alpha=float(entry.get("alpha", 90)),
                    beta=float(entry.get("beta", 90)),
                    gamma=float(entry.get("gamma", 90)),
                    cif_url=f"{self.BASE_URL}/{cod_id}.cif",
                ))
            except (ValueError, TypeError):
                continue
        return structures

    def download_cif(self, cod_id: str, out_dir: str) -> str:
        """Download a CIF file from COD.

        Returns path to saved CIF file.
        """
        out_path = Path(out_dir) / f"{cod_id}.cif"
        if out_path.exists():
            return str(out_path)

        self._rate_limit()
        url = f"{self.BASE_URL}/{cod_id}.cif"
        req = Request(url, headers={"User-Agent": "ZynerjiChirality/0.5.0"})
        try:
            with urlopen(req, timeout=30) as resp:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(resp.read())
            return str(out_path)
        except (URLError, HTTPError) as e:
            logger.warning("Failed to download CIF %s: %s", cod_id, e)
            return ""

    @staticmethod
    def save_structures(structures: list[CrystalStructure], path: str) -> None:
        """Save structures to JSON checkpoint."""
        data = [s.to_dict() for s in structures]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info("Saved %d structures to %s", len(data), path)

    @staticmethod
    def load_structures(path: str) -> list[CrystalStructure]:
        """Load structures from JSON checkpoint."""
        with open(path) as f:
            data = json.load(f)
        return [CrystalStructure.from_dict(d) for d in data]
