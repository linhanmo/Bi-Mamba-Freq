import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def create_experiment_dir(root: Path, task_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = root / task_name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def to_jsonable(value: Any) -> Any:
    return _to_jsonable(value)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2))
