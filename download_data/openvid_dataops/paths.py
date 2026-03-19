from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataLayout:
    root: Path

    @property
    def data_root(self) -> Path:
        return self.root / "data" / "openvid"

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def zip_root(self) -> Path:
        return self.raw_root / "zips"

    @property
    def parts_root(self) -> Path:
        return self.raw_root / "parts"

    @property
    def csv_path(self) -> Path:
        return self.raw_root / "OpenVid-1M.csv"

    @property
    def manifests_root(self) -> Path:
        return self.data_root / "manifests"

    @property
    def encoded_root(self) -> Path:
        return self.data_root / "encoded"

    @property
    def state_root(self) -> Path:
        return self.data_root / "state"

    @property
    def logs_root(self) -> Path:
        return self.data_root / "logs"


def ensure_layout(layout: DataLayout) -> None:
    for p in [
        layout.data_root,
        layout.raw_root,
        layout.zip_root,
        layout.parts_root,
        layout.manifests_root,
        layout.encoded_root,
        layout.state_root,
        layout.logs_root,
    ]:
        p.mkdir(parents=True, exist_ok=True)
