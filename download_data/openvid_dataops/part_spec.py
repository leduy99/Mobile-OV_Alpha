import ast
from typing import Iterable, List, Optional


def parse_parts_spec(spec: str, allow_all: bool = False) -> Optional[List[int]]:
    """
    Parse part spec from formats like:
    - "[1,2,4,5]"
    - "1,2,4,5"
    - "1"
    - "all" (if allow_all=True)
    """
    if not spec:
        raise ValueError("parts spec is empty")

    value = spec.strip()
    if allow_all and value.lower() == "all":
        return None

    if value.startswith("["):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, int):
            parts = [parsed]
        elif isinstance(parsed, (list, tuple)):
            parts = list(parsed)
        else:
            raise ValueError(f"unsupported parts format: {type(parsed)}")
    else:
        parts = [int(x.strip()) for x in value.split(",") if x.strip()]

    if not parts:
        raise ValueError("no valid parts parsed")

    out = sorted({int(p) for p in parts})
    for p in out:
        if p < 0:
            raise ValueError(f"part must be >= 0, got {p}")
    return out


def parts_to_remote_ids(parts: Iterable[int], part_index_base: int) -> List[int]:
    if part_index_base not in (0, 1):
        raise ValueError("part_index_base must be 0 or 1")
    remote = [int(p) - part_index_base for p in parts]
    for rid in remote:
        if rid < 0:
            raise ValueError(
                f"invalid part after base conversion: remote_id={rid}. "
                "Check --part-index-base and --parts"
            )
    return remote
