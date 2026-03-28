#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


SEARCH_ROOT = Path(__file__).resolve().parent.parent
QUEUE_ROOT = SEARCH_ROOT / "queue"
FAILED_DIR = QUEUE_ROOT / "failed"
LOGS_DIR = QUEUE_ROOT / "logs"
STAGE_ROOT = SEARCH_ROOT / "staging" / "gauss-index-batch-20260328"


def load_manifest(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_candidates() -> Dict[str, Dict[str, Any]]:
    import sys

    if str(SEARCH_ROOT) not in sys.path:
        sys.path.insert(0, str(SEARCH_ROOT))
    from index_200_sweep import make_candidates  # noqa: E402

    return {candidate.label: candidate.__dict__ for candidate in make_candidates()}


def classify_stage_manifests() -> Dict[str, Any]:
    candidate_specs = load_candidates()
    manifest_paths = sorted((STAGE_ROOT / "manifests").glob("*/*.json"))

    owner_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    failure_buckets: Counter[str] = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)

    for path in manifest_paths:
        payload = load_manifest(path)
        owner_counts[payload["owner"]] += 1
        shape_name = path.parent.name
        shape_counts[shape_name] += 1
        candidate_label = path.stem
        spec = candidate_specs[candidate_label]

        if spec["dot_variant"] == "broadcast_reduce" and shape_name != "decode":
            bucket = "broadcast_reduce_prefill_shape_mismatch"
            failure_buckets[bucket] += 1
            if len(examples[bucket]) < 5:
                examples[bucket].append(str(path.relative_to(SEARCH_ROOT)))

    if len(owner_counts) != 1 or next(iter(owner_counts.keys())) != "gauss":
        bucket = "owner_metadata_not_gauss"
        failure_buckets[bucket] = len(manifest_paths)
        for path in manifest_paths[:5]:
            examples[bucket].append(str(path.relative_to(SEARCH_ROOT)))

    return {
        "manifest_count": len(manifest_paths),
        "shape_counts": dict(shape_counts),
        "owner_counts": dict(owner_counts),
        "failure_buckets": dict(failure_buckets),
        "examples": dict(examples),
    }


def queue_state() -> Dict[str, Any]:
    failed = sorted(FAILED_DIR.glob("*.json"))
    logs = sorted(LOGS_DIR.glob("*.log"))
    return {
        "failed_manifest_count": len(failed),
        "failed_manifests": [str(path.relative_to(SEARCH_ROOT)) for path in failed[:10]],
        "log_count": len(logs),
        "logs": [str(path.relative_to(SEARCH_ROOT)) for path in logs[:10]],
    }


def main() -> None:
    payload = {
        "queue_state": queue_state(),
        "stage_state": classify_stage_manifests(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
