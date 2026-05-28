"""Logger para notebooks: imprime en la celda y guarda en `<owner>/logs/<juego>/<notebook>.log`.

El `<juego>` se deriva del `notebook_name` lowercaseado (`MP`→`mp`, `BoS`→`bos`,
`ThreePlayers`→`threeplayers`), siguiendo la misma convención por juego que
usan `images/` y `data/`.

Uso típico desde una notebook (en `entrega/`, `fernando/`, `german/` o `sebastian/`):

    import sys
    sys.path.append("..")
    from utils.logger import get_logger

    log = get_logger()
    log.info("entrenamiento iniciado")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent  # 2 niveles arriba de utils/
_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _detect_notebook_path() -> Path | None:
    """Devuelve la ruta del .ipynb en ejecución, o None si no se puede determinar."""
    try:
        import IPython

        ip = IPython.get_ipython()
        if ip is not None:
            for key in ("__vsc_ipynb_file__", "__session__"):
                value = ip.user_ns.get(key)
                if value:
                    return Path(value).resolve()
    except Exception:
        pass

    try:
        import ipynbname

        return Path(ipynbname.path()).resolve()
    except Exception:
        return None


def _resolve_owner_and_name(
    notebook_path: Path | None, notebook_name: str | None
) -> tuple[Path, str]:
    """En esta entrega el owner es siempre `_REPO_ROOT`."""
    name = notebook_name
    if name is None and notebook_path is not None:
        name = notebook_path.stem
    if name is None:
        raise RuntimeError(
            "No se pudo detectar el nombre del notebook automáticamente. "
            "Pasá `notebook_name` explícitamente: get_logger(notebook_name='mi_notebook')."
        )
    return _REPO_ROOT, name


def get_logger(
    notebook_name: str | None = None,
    level: int = logging.INFO,
    game: str | None = None,
) -> logging.Logger:
    """Logger con doble salida: stdout (visible en la notebook) y archivo en `<owner>/logs/<game>/`.

    - `notebook_name`: opcional. Si no se pasa, se detecta desde la ruta del notebook.
    - `level`: nivel mínimo (por defecto INFO).
    - `game`: subdirectorio dentro de `logs/`. Si no se pasa, se deriva del
      `notebook_name` lowercaseado (`MP`→`mp`, `BoS`→`bos`).
    """
    notebook_path = _detect_notebook_path()
    owner_dir, name = _resolve_owner_and_name(notebook_path, notebook_name)

    if game is None:
        game = name.lower()
    logs_dir = owner_dir / "logs" / game
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{name}.log"

    logger = logging.getLogger(f"notebook.{owner_dir.name}.{name}")
    logger.setLevel(level)
    logger.propagate = False

    # Re-ejecutar la celda no debe duplicar handlers
    has_file = any(
        isinstance(h, logging.FileHandler)
        and Path(h.baseFilename).resolve() == log_file.resolve()
        for h in logger.handlers
    )
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )

    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT)

    if not has_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if not has_stream:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger
