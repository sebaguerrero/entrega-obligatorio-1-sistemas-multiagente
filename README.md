# Obligatorio 1 — Sistemas Multiagente

Entrega del primer obligatorio de la materia Sistemas Multiagente. El proyecto implementa y evalúa algoritmos de aprendizaje multiagente sobre 8 juegos simultáneos, analizando convergencia a equilibrios de Nash en distintos escenarios (poblaciones homogéneas, mixtas, contra rivales no estratégicos).

**Integrantes**: Sebastián Guerrero (158947), Germán López (164030), Fernando Zanetti (167581).

## En qué consiste

Cada juego se estudia bajo varios pairings de agentes:

- **Self-play homogéneo**: dos o tres copias del mismo algoritmo (FP×n, RM×n).
- **Cross-play**: FP contra RM.
- **Contra Random**: aprendiz vs `RandomAgent` (uniforme sobre acciones), para evaluar BR a una distribución uniforme.
- **Mixto**: combinaciones FP + RM + Random en juegos de 3 jugadores.

Para cada experimento se persiste el `history` completo (5000 iteraciones) como pickle en `data/<juego>/`, las gráficas en `images/<juego>/`, y los logs en `logs/<juego>/` cuando se ejecutan los notebooks.

**Algoritmos implementados**:

- **Fictitious Play (FP)** — mejor respuesta a la frecuencia empírica de los rivales.
- **Regret Matching (RM)** — política proporcional al regret acumulado positivo.
- **RandomAgent** — política uniforme constante (baseline no estratégico).
- **Independent Q-Learning (IQL)** — Q-learning tratando al resto del entorno como estacionario; usado en Foraging.
- **Joint Action Learning with Agent Modelling (JAL-AM)** — Q-learning sobre acciones conjuntas con modelo del rival; usado en Foraging.

**Juegos implementados** (uno por notebook):

| Notebook | Juego |
|---|---|
| `MP.ipynb` | Matching Pennies |
| `RPS.ipynb` | Rock-Paper-Scissors |
| `Blotto.ipynb` | Coronel Blotto |
| `BoS.ipynb` | Battle of the Sexes |
| `Chicken.ipynb` | Chicken |
| `Cournot.ipynb` | Cournot (duopolio y triopolio) |
| `ThreePlayers.ipynb` | Juegos de 3 jugadores (Bonanno, Aumann) |
| `Foraging.ipynb` | Level-Based Foraging (entorno secuencial) |

## Estructura del proyecto

```
entrega-obligatorio-1-sistemas-multiagente/
├── notebooks/              # 8 notebooks, uno por juego
├── agents/                 # Implementaciones de algoritmos
├── base/                   # Clases abstractas (Agent, Game)
├── games/                  # 8 juegos
├── utils/                  # Logger, plots, storage, runner
├── data/                   # Pickles cacheados de history dicts (5000 iter)
├── images/                 # PNGs generados por los notebooks
├── informe/                # Informe en PDF
├── pyproject.toml          # Dependencias y metadata
├── uv.lock                 # Versiones exactas para reproducibilidad
├── .python-version         # Versión de Python (3.11.11)
├── .gitignore
└── README.md
```

## Instalación

El proyecto requiere **Python 3.11.x** (ver `.python-version`) y está configurado con [uv](https://docs.astral.sh/uv/) como package manager.

### Opción A — con uv (recomendado)

```bash
# Instalar uv si no está disponible
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sincronizar dependencias (crea .venv/ con las versiones del uv.lock)
uv sync

# Activar el entorno
source .venv/bin/activate
```

### Opción B — con pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Cómo correr los notebooks

```bash
# Con uv
uv run jupyter lab notebooks/

# O con el entorno activado
jupyter lab notebooks/
```

Cada notebook se puede ejecutar de principio a fin. Si los pickles ya están en `data/<juego>/` (es el caso en este repositorio), `exp.cached_run(..., force=False)` los reutiliza; con `force=True` (default en algunas celdas) re-ejecuta el experimento y sobreescribe el cache.

Las gráficas se guardan automáticamente en `images/<juego>/`. Los logs en `logs/<juego>/<notebook>.log`.

## Contenido por carpeta

### `notebooks/`

Los 8 notebooks de la entrega. Cada uno arranca con `sys.path.append('..')` para acceder a `utils/`, `games/`, `agents/` desde el directorio padre.

### `agents/`

Implementaciones de los algoritmos. Todos heredan de `base.Agent` y exponen `action()` (muestrea acción), `update(observation)` (incorpora la observación de la ronda anterior), y `policy()` (devuelve la política aprendida).

- `fictitiousplay.py` — FP: mejor respuesta a la frecuencia empírica del rival.
- `regretmatching.py` — RM: política proporcional a `max(0, regret acumulado)`.
- `random_agent.py` — RandomAgent: muestreo uniforme sobre acciones.
- `iql_agent.py` — IQL: Q-learning independiente con ε-greedy.
- `jal_am_agent.py` — JAL-AM: Q-learning sobre acciones conjuntas + modelo del rival.

### `base/`

Clases abstractas que definen el contrato de agentes y juegos.

- `agent.py` — `Agent`: interfaz `action() / update() / policy()`.
- `game.py` — `Game`: interfaz compatible con `pettingzoo.ParallelEnv` (`reset()`, `step()`, espacios de acción y observación).

### `games/`

Cada archivo implementa un juego como subclase de `Game`. Los juegos one-shot precargan matrices de payoffs; los iterados (Foraging) envuelven entornos externos.

- `mp.py`, `rps.py`, `blotto.py`, `bos.py`, `chicken.py`, `cournot.py`, `threeplayers.py`, `foraging.py`.

### `utils/`

Infraestructura compartida por todos los notebooks.

- `experiments.py` — `run_demo` (loop de simulación) y `cached_run` (versión cacheada en disco).
- `plots.py` — funciones de gráficas reutilizables: `plot_policy_evolution`, `plot_cumulative_regret`, `plot_action_over_time`, `plot_reward_over_time`, `plot_distance_to_nash`, `plot_policy_bars`, entre otras.
- `storage.py` — resuelve paths de `data/`, `images/` y `logs/` relativos al root del proyecto.
- `logger.py` — logger con doble salida (stdout + archivo en `logs/<juego>/<notebook>.log`).

### `data/`

Pickles con `history` dicts persistidos por `exp.cached_run`. Cada archivo contiene `policy_history`, `action_history`, `reward_history`, `curr_policy_history`, y `cum_regrets_history` (cuando aplica) para todos los agentes a lo largo de 5000 iteraciones.

Estructura: `data/<juego>/<nombre_experimento>.pkl`.

**Nota sobre Foraging**: en algunos escenarios con agentes JAL-AM, el pickle de agentes entrenados (`agents_foraging_*.pkl`, que contiene las Q-tables) supera los límites de tamaño y no se incluye en el repositorio. Se puede regenerar ejecutando la celda de entrenamiento correspondiente en `notebooks/Foraging.ipynb`.

### `images/`

PNGs generados por los notebooks. Estructura: `images/<juego>/<nombre_plot>.png`. Se sobrescriben cada vez que se ejecutan las celdas correspondientes.

### `informe/`

`informe.pdf` — documento final con el análisis de los resultados, fórmulas teóricas, gráficas y conclusiones por juego.

## Repositorio

<https://github.com/sebaguerrero/entrega-obligatorio-1-sistemas-multiagente>
