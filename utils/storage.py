"""Helpers de persistencia compartidos por las notebooks del equipo.

Provee tres familias de utilidades, todas resolviendo automáticamente la
carpeta del owner (`fernando`, `sebastian`, `german`, `entrega`) de la misma
forma que `utils.logger`:

1. **Cache de DataFrames / objetos arbitrarios** (`cached`) — pickle a
   `<owner>/data/<juego>/<nombre>.pkl`. Útil para no re-correr experimentos
   lentos cuando se re-abre el notebook. El `<juego>` se deriva del prefijo
   del `name` (split por `_`); se puede sobrescribir con `game=`.

2. **Persistencia de agentes entrenados** (`save_agents` / `load_agents`) —
   pickle desacoplando `self.game` antes de serializar (los envs de
   gymnasium no son trivialmente picklables). Mismo layout
   `<owner>/data/<juego>/agents_<nombre>.pkl`.

3. **Guardado automático de figuras** (`save_fig` / `savefig`) — escribe en
   `<owner>/images/<juego>/<nombre>.png` con DPI 120.

Uso típico desde una notebook:

    from utils.storage import cached, savefig, save_agents, load_agents

    df = cached('mp_oneshot', lambda: run_experiment(...))  # → data/mp/mp_oneshot.pkl
    plt.plot(df['x'], df['y'])
    savefig('mp/mi_metrica')
    plt.show()

"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent  # 2 niveles arriba de utils/

def resolve_owner(owner: str | Path | None = None) -> Path:
    """Carpeta raíz del proyecto (donde vive `utils/`).

    En esta entrega solo existe un único root. Si se pasa `owner` como `Path`
    explícito se respeta; en cualquier otro caso devuelve `_REPO_ROOT`.
    """
    if isinstance(owner, Path):
        return owner.resolve()
    return _REPO_ROOT


def _derive_game(name: str) -> str:
    """Deriva el juego del prefijo del `name` (split por `_`).

    Convención: los nombres siempre arrancan con el juego, p.ej.
    `mp_oneshot`, `blotto_history_fp_vs_rm`, `foraging_5x5_iql`. Si el `name`
    no tiene `_`, se usa todo el `name` como juego.
    """
    return name.split("_", 1)[0]


def data_dir(game: str | None = None,
             owner: str | Path | None = None) -> Path:
    """Devuelve `<owner>/data/[<game>/]` creándolo si no existe.

    Si `game` es None devuelve el root `<owner>/data/`.
    """
    d = resolve_owner(owner) / "data"
    if game is not None:
        d = d / game
    d.mkdir(parents=True, exist_ok=True)
    return d


def images_dir(owner: str | Path | None = None) -> Path:
    """Devuelve `<owner>/images/` creándolo si no existe."""
    d = resolve_owner(owner) / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─── Cache genérico ───────────────────────────────────────────────────────────

def cached(name: str, fn: Callable, force: bool = False,
           owner: str | Path | None = None, game: str | None = None):
    """Cachea el resultado de `fn()` en disco con la clave `name`.

    Path resultante: `<owner>/data/<game>/<name>.pkl`. Si `game` no se pasa,
    se deriva del prefijo del `name` (split por `_`). Si el archivo existe y
    `force=False`, se levanta; si no, se ejecuta `fn` y se guarda el resultado.
    """
    if game is None:
        game = _derive_game(name)
    path = data_dir(game, owner) / f"{name}.pkl"
    if path.exists() and not force:
        with open(path, "rb") as f:
            return pickle.load(f)
    result = fn()
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


# ─── Persistencia de agentes entrenados ───────────────────────────────────────

def save_agents(agents: dict, name: str,
                owner: str | Path | None = None,
                game: str | None = None) -> Path:
    """Serializa un dict de agentes entrenados a `<owner>/data/<game>/agents_{name}.pkl`.

    Desacopla `self.game` de cada agente antes de picklear (los juegos pueden
    tener handles no serializables — envs de gymnasium con file descriptors,
    threadlocks, etc.). El estado interno del agente (`q`, `counts`, `t`,
    `cum_regrets`, `learned_policy`, RNG, etc.) sí se preserva.

    Para cargar de vuelta usar `load_agents(name, game)` pasando una instancia
    fresca del mismo juego.
    """
    if game is None:
        game = _derive_game(name)
    path = data_dir(game, owner) / f"agents_{name}.pkl"
    saved_games = {a: agents[a].game for a in agents}
    try:
        for a in agents:
            agents[a].game = None
        with open(path, "wb") as f:
            pickle.dump(agents, f)
    finally:
        for a in agents:
            agents[a].game = saved_games[a]
    return path


def load_agents(name: str, game, owner: str | Path | None = None,
                game_folder: str | None = None) -> dict:
    """Carga agentes desde `<owner>/data/<game_folder>/agents_{name}.pkl` y re-conecta `game`.

    `game` es la instancia del juego que se re-engancha a cada agente.
    `game_folder` es el subdirectorio (por convención el nombre del juego);
    si no se pasa, se deriva del prefijo del `name`.

    Devuelve dict[agent_id → instancia] listo para usar (típicamente con
    `agent.learn = False` para evaluación greedy).
    """
    if game_folder is None:
        game_folder = _derive_game(name)
    path = data_dir(game_folder, owner) / f"agents_{name}.pkl"
    with open(path, "rb") as f:
        agents = pickle.load(f)
    for a in agents:
        agents[a].game = game
    return agents


# ─── Guardado de figuras ──────────────────────────────────────────────────────

def save_fig(fig, save: str | None, owner: str | Path | None = None):
    """Guarda `fig` en `<owner>/images/{game}/{name}.png` si `save` tiene
    formato `'game/name'`. Si `save` es None o fig es None, no hace nada.

    Helper interno usado por las funciones de plot de `experiments.py`.
    """
    if save is None or fig is None:
        return
    if "/" not in save:
        raise ValueError(f"save debe tener formato 'game/name', recibí: {save!r}")
    game, name = save.split("/", 1)
    target_dir = images_dir(owner) / game
    target_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_dir / f"{name}.png", dpi=120, bbox_inches="tight")


def savefig(name: str, owner: str | Path | None = None):
    """Guarda la figura actual (`plt.gcf()`) en `<owner>/images/{game}/{name}.png`.

    Útil cuando el plot se construye en el notebook con `plt.subplots(...)`
    y `plt`/`ax` directos (no a través de una función `plot_xxx`). Llamar
    antes de `plt.show()`. `name` debe tener formato `'game/descriptive_name'`.
    """
    import matplotlib.pyplot as plt
    save_fig(plt.gcf(), name, owner)
