import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def create_experiment_dir(root: Path, task_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = root / task_name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

