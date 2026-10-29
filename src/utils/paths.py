"""Resolucion de rutas locales desde configs/paths.yaml (datos fuera del repo)."""
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"

load_dotenv(REPO_ROOT / ".env")


def _resolve(node, data_root):
    if isinstance(node, dict):
        return {k: _resolve(v, data_root) for k, v in node.items()}
    if isinstance(node, str):
        return Path(re.sub(r"\$\{data_root\}", data_root, node))
    return node


def load_paths():
    """Devuelve el arbol de rutas como objetos Path.

    INSPECTAI_DATA_ROOT (en .env) tiene prioridad sobre data_root del yaml.
    """
    path_file = CONFIGS / "paths.yaml"
    if not path_file.exists():
        raise FileNotFoundError(
            "Falta configs/paths.yaml: copia configs/paths.example.yaml y ajusta las rutas."
        )
    cfg = yaml.safe_load(path_file.read_text(encoding="utf-8"))
    data_root = os.getenv("INSPECTAI_DATA_ROOT", cfg.pop("data_root"))
    paths = _resolve(cfg, data_root)
    paths["data_root"] = Path(data_root)
    return paths


def load_training_config():
    return yaml.safe_load((CONFIGS / "training_config.yaml").read_text(encoding="utf-8"))
