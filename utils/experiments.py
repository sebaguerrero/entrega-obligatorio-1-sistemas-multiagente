"""Helpers para los experimentos del informe.

Provee funciones para:
  - Correr experimentos con múltiples seeds sobre juegos one-shot y secuenciales.
  - Calcular métricas estándar (distancia a Nash, entropía, distribución de joint actions).
  - Plotear con bandas de confianza vía seaborn.

Convención: todas las funciones de simulación devuelven DataFrames en formato LARGO
(columnas `seed`, `iter`/`episode`, `agent`, `metric`, `value`) para que sean fáciles
de graficar con seaborn.

Las utilidades genéricas de persistencia (data pickle, save/load agents,
save_fig, savefig) viven en `utils.storage` — se re-exportan acá para no
romper código existente.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

# Re-export de utilidades genéricas desde utils.storage para que el código
# de los notebooks siga funcionando sin cambios (exp.cached, exp.savefig, etc.).
from utils.storage import (
    cached,
    save_agents,
    load_agents,
    savefig,
)

# Re-export de plot helpers desde utils.plots — los notebooks usan
# `exp.plot_policy_evolution(...)`, `exp.plot_joint_action_heatmap(...)`, etc.
from utils.plots import (
    plot_policy_evolution,
    plot_policy_heatmap,
    plot_expected_value_evolution,
    plot_distance_to_nash,
    plot_cumulative_average_reward,
    plot_cumulative_regret,
    plot_average_positive_regret,
    plot_instantaneous_vs_average_policy,
    plot_simplex_pairs,
    plot_qtable_growth,
    plot_metric_with_band,
    plot_simplex_2d,
    plot_utility_space,
    plot_joint_action_heatmap,
    plot_action_histogram,
    plot_actions_scatter,
    plot_action_over_time,
    plot_reward_over_time,
    plot_policy_bars,
    plot_expected_value_instantaneous_vs_average,
    plot_cournot_best_response,
    plot_foraging_rewards,
    plot_foraging_policy_stability,
    plot_foraging_states_qtable,
    plot_foraging_coordination,
)

# ─── Simulación one-shot ───────────────────────────────────────────────────────


def play_oneshot(game, agents: dict, iterations: int) -> dict:
    """Corre `iterations` rondas de un juego one-shot.

    Devuelve un dict con:
      - `policy_history`: dict[agent → ndarray (iterations, n_actions)] con la
        política aprendida en cada iteración (lo que devuelve `policy()`).
      - `action_history`: dict[agent → ndarray (iterations,)] con la acción jugada.
      - `reward_history`: dict[agent → ndarray (iterations,)] con el reward del paso.
      - `curr_policy_history`: dict[agent → ndarray] sólo para agentes que tengan
        atributo `curr_policy` (RM). Captura la política instantánea por iteración.
      - `cum_regrets_history`: dict[agent → ndarray] sólo para agentes que tengan
        atributo `cum_regrets` (RM). Captura los regrets acumulados.
    """
    g = game
    g.reset()
    n_actions = {a: g.num_actions(a) for a in g.agents}

    policy_history = {a: np.zeros((iterations, n_actions[a])) for a in g.agents}
    action_history = {a: np.zeros(iterations, dtype=int) for a in g.agents}
    reward_history = {a: np.zeros(iterations) for a in g.agents}

    # Captura opcional para agentes RM (que tienen estos atributos)
    has_curr = {a: hasattr(agents[a], "curr_policy") for a in g.agents}
    has_regrets = {a: hasattr(agents[a], "cum_regrets") for a in g.agents}
    curr_policy_history = {
        a: np.zeros((iterations, n_actions[a])) for a in g.agents if has_curr[a]
    }
    cum_regrets_history = {
        a: np.zeros((iterations, n_actions[a])) for a in g.agents if has_regrets[a]
    }

    for t in range(iterations):
        actions = {a: int(agents[a].action()) for a in g.agents}
        g.step(actions)
        for a in g.agents:
            action_history[a][t] = actions[a]
            reward_history[a][t] = g.reward(a)
            pol = agents[a].policy()
            policy_history[a][t, : len(pol)] = pol
            if has_curr[a]:
                cp = agents[a].curr_policy
                curr_policy_history[a][t, : len(cp)] = cp
            if has_regrets[a]:
                cr = agents[a].cum_regrets
                cum_regrets_history[a][t, : len(cr)] = cr

    return {
        "policy_history": policy_history,
        "action_history": action_history,
        "reward_history": reward_history,
        "curr_policy_history": curr_policy_history,
        "cum_regrets_history": cum_regrets_history,
    }


def run_oneshot(
    game_factory: Callable,
    agent_pair: list,
    seeds: list[int],
    iterations: int,
    nash: dict | None = None,
    log=None,
    log_label: str | None = None,
) -> pd.DataFrame:
    """Corre un juego one-shot con cada seed y devuelve un DataFrame largo.

    Parameters
    ----------
    game_factory : Callable
        Función que devuelve una instancia fresca del juego (ej. `lambda: RPS()`).
    agent_pair : list of (AgentClass, kwargs) tuples
        Una entrada por agente. `kwargs` puede incluir `initial` etc.
        El `seed` se inyecta automáticamente desde `seeds`.
    seeds : list[int]
        Lista de seeds para correr.
    iterations : int
        Iteraciones por seed.
    nash : dict, optional
        Si se da, dict[agent → ndarray] con la política de Nash teórica.
        Se computa la distancia euclídea en cada iteración.
    log : logging.Logger, optional
        Si se pasa, se loguea inicio/fin del experimento y por seed.
    log_label : str, optional
        Etiqueta legible para los mensajes de log (ej. "FP vs RM en RPS").

    Returns
    -------
    pd.DataFrame con columnas `seed`, `iter`, `agent`, `metric`, `value`.
    Métricas incluidas: `policy_a0..a_n` (una por acción), `reward_cum`,
    `dist_to_nash` (si nash != None).
    """
    label = log_label or "one-shot"
    if log is not None:
        log.info(f"[{label}] inicio: {len(seeds)} seeds × {iterations} iter")
    t_start = time.time()
    rows = []
    for seed in seeds:
        t_seed = time.time()
        g = game_factory()
        g.reset()
        agents = {}
        for idx, agent_name in enumerate(g.agents):
            AgentClass, kwargs = agent_pair[idx]
            kw = dict(kwargs)
            kw.setdefault("seed", seed + idx)  # seeds distintos por agente
            agents[agent_name] = AgentClass(game=g, agent=agent_name, **kw)

        result = play_oneshot(g, agents, iterations)
        cum_reward = {a: np.cumsum(result["reward_history"][a]) for a in g.agents}

        for t in range(iterations):
            for a in g.agents:
                pol = result["policy_history"][a][t]
                # Probabilidades por acción
                for k, p in enumerate(pol):
                    rows.append((seed, t, a, f"policy_a{k}", float(p)))
                # Reward promedio acumulado
                rows.append(
                    (seed, t, a, "reward_cum_avg", float(cum_reward[a][t] / (t + 1)))
                )
                # Distancia al Nash
                if nash is not None:
                    nash_a = nash[a] if a in nash else nash.get("default")
                    if nash_a is not None:
                        dist = float(np.linalg.norm(pol[: len(nash_a)] - nash_a))
                        rows.append((seed, t, a, "dist_to_nash", dist))
                # Entropía
                p = pol[pol > 0]
                entropy = float(-np.sum(p * np.log(p))) if len(p) > 0 else 0.0
                rows.append((seed, t, a, "entropy", entropy))
        if log is not None:
            elapsed = time.time() - t_seed
            log.info(f"[{label}] seed={seed} ok ({elapsed:.1f}s)")

    if log is not None:
        total = time.time() - t_start
        log.info(f"[{label}] fin ({total:.1f}s total)")
    return pd.DataFrame(rows, columns=["seed", "iter", "agent", "metric", "value"])


# ─── Simulación secuencial (Foraging) ──────────────────────────────────────────


def play_episode_sequential(game, agents, max_steps: int = 1000) -> dict[str, float]:
    """Corre un episodio completo de un juego secuencial hasta `done()`.

    Devuelve dict[agent → cum_reward] del episodio.
    """
    game.reset()
    for a in game.agents:
        if hasattr(agents[a], "reset"):
            agents[a].reset()
    cum = {a: 0.0 for a in game.agents}
    step = 0
    while not game.done() and step < max_steps:
        acts = {a: agents[a].action() for a in game.agents}
        game.step(acts)
        for a in game.agents:
            cum[a] += game.reward(a)
            if hasattr(agents[a], "update"):
                agents[a].update()
        step += 1
    return cum


def run_sequential(
    game_factory: Callable,
    agent_pair: list,
    seeds: list[int],
    episodes_per_iter: int,
    iterations: int,
    eval_episodes: int = 0,
    log=None,
    log_label: str | None = None,
    log_every: int | None = None,
) -> pd.DataFrame:
    """Entrena agentes en un juego secuencial sobre múltiples seeds.

    Parameters
    ----------
    game_factory : Callable
        Función que devuelve una instancia fresca del juego.
    agent_pair : list of (AgentClass, config) tuples
        Una entrada por agente. `config` es la instancia de Config del agente.
        El `seed` se inyecta automáticamente.
    seeds : list[int]
        Seeds a usar.
    episodes_per_iter : int
        Episodios que se agregan en una "iteración" (un punto de la curva).
    iterations : int
        Cantidad de puntos en la curva de aprendizaje.
    eval_episodes : int
        Si > 0, después del entrenamiento corre `eval_episodes` episodios con
        `learn=False` y registra el reward promedio greedy.
    log : logging.Logger, optional
        Logger para registrar inicio/fin del entrenamiento, progreso por seed
        y evaluación final.
    log_label : str, optional
        Etiqueta legible (ej. "IQL en Foraging-5x5-2p"). Default: "sequential".
    log_every : int, optional
        Cada cuántas iteraciones loguear el reward promedio actual (default:
        `max(1, iterations // 10)`).

    Returns
    -------
    pd.DataFrame con columnas `seed`, `iter`, `agent`, `metric`, `value`.
    Métricas: `reward_train` (durante el aprendizaje), `reward_eval` (greedy),
    `q_size` (tamaño de la tabla Q).
    """
    label = log_label or "sequential"
    if log_every is None:
        log_every = max(1, iterations // 10)
    if log is not None:
        log.info(
            f"[{label}] inicio: {len(seeds)} seeds × {iterations} iter × "
            f"{episodes_per_iter} eps + {eval_episodes} eps eval"
        )
    t_start = time.time()
    rows = []
    for seed in seeds:
        t_seed = time.time()
        if log is not None:
            log.info(f"[{label}] seed={seed} — entrenando...")
        game = game_factory(seed)
        agents = {}
        for idx, agent_name in enumerate(game.agents):
            AgentClass, config_factory = agent_pair[idx]
            # Inyectar seed en config si tiene atributo seed
            cfg = config_factory(seed + idx)
            agents[agent_name] = AgentClass(game=game, agent=agent_name, config=cfg)

        # Entrenamiento
        for it in range(iterations):
            ep_rewards = {a: 0.0 for a in game.agents}
            for _ in range(episodes_per_iter):
                cum = play_episode_sequential(game, agents)
                for a in game.agents:
                    ep_rewards[a] += cum[a]
            for a in game.agents:
                avg = ep_rewards[a] / episodes_per_iter
                rows.append((seed, it, a, "reward_train", float(avg)))
                if hasattr(agents[a], "q"):
                    rows.append((seed, it, a, "q_size", float(len(agents[a].q))))
            if log is not None and (it + 1) % log_every == 0:
                avgs = {
                    a: round(ep_rewards[a] / episodes_per_iter, 3) for a in game.agents
                }
                q_sizes = {
                    a: len(agents[a].q) for a in game.agents if hasattr(agents[a], "q")
                }
                log.info(
                    f"[{label}] seed={seed} iter {it + 1}/{iterations} reward={avgs} |q|={q_sizes}"
                )

        # Evaluación greedy
        if eval_episodes > 0:
            for a in game.agents:
                if hasattr(agents[a], "learn"):
                    agents[a].learn = False
            eval_totals = {a: 0.0 for a in game.agents}
            eval_success = 0
            for _ in range(eval_episodes):
                cum = play_episode_sequential(game, agents)
                for a in game.agents:
                    eval_totals[a] += cum[a]
                # Episodio exitoso = alguien cosechó al menos una fruta
                if any(cum[a] > 0 for a in game.agents):
                    eval_success += 1
            if log is not None:
                eval_avg = {
                    a: round(eval_totals[a] / eval_episodes, 3) for a in game.agents
                }
                log.info(
                    f"[{label}] seed={seed} eval (learn=False) → reward={eval_avg}, "
                    f"success_rate={eval_success / eval_episodes:.3f}"
                )
            for a in game.agents:
                avg = eval_totals[a] / eval_episodes
                rows.append((seed, iterations, a, "reward_eval", float(avg)))
            rows.append(
                (
                    seed,
                    iterations,
                    "_team",
                    "success_rate",
                    float(eval_success / eval_episodes),
                )
            )

        if log is not None:
            elapsed = time.time() - t_seed
            log.info(f"[{label}] seed={seed} fin ({elapsed:.1f}s)")

    if log is not None:
        total = time.time() - t_start
        log.info(f"[{label}] fin total ({total:.1f}s)")
    return pd.DataFrame(rows, columns=["seed", "iter", "agent", "metric", "value"])


# ─── Demos didácticas para notebooks por juego ────────────────────────────────


def run_demo(game, agent_pair: list, iterations: int, seed: int = 1):
    """Corrida ilustrativa one-shot para notebooks didácticos.

    Crea los agentes según `agent_pair` (cada entrada = (AgentClass, kwargs)),
    inyecta `seed + idx` en cada uno y juega `iterations` rondas.

    Devuelve `(history, agents)`:
      - `history`: dict con `policy_history`, `action_history`, `reward_history`.
      - `agents`: dict[agent_name → instancia] por si el notebook quiere
        inspeccionar estado interno (count, regrets, etc.).
    """
    game.reset()
    agents = {}
    for idx, agent_name in enumerate(game.agents):
        AgentClass, kw = agent_pair[idx]
        kw = dict(kw)
        kw.setdefault("seed", seed + idx)
        agents[agent_name] = AgentClass(game=game, agent=agent_name, **kw)
    history = play_oneshot(game, agents, iterations)
    return history, agents


def cached_run(
    name: str,
    game,
    agent_pair: list,
    iterations: int,
    seed: int = 1,
    force: bool = True,
    owner: str | None = None,
    game_folder: str | None = None,
) -> dict:
    """Versión cacheada de `run_demo`: persiste el `history` dict en
    `<owner>/data/<juego>/<name>.pkl` para no re-correr el experimento cada vez.

    Devuelve **sólo el `history`**, no los `agents` (sus referencias a
    `self.game` no son trivialmente picklables; si necesitás el agente
    entrenado para evaluar después, usar `save_agents()` aparte).

    El `<juego>` se deriva del prefijo del `name` (split por `_`); se puede
    sobrescribir con `game_folder=`.

    Por defecto **siempre re-ejecuta** (`force=True`) y sobrescribe el cache.
    Esto evita que cambios en el código de los agentes queden enmascarados
    por un pickle viejo. Pasar `force=False` explícitamente para reutilizar
    el cache cuando el experimento es costoso y no cambió.
    """

    def _run():
        history, _agents = run_demo(game, agent_pair, iterations, seed=seed)
        return history

    return cached(name, _run, force=force, owner=owner, game=game_folder)


# ─── Métricas analíticas ───────────────────────────────────────────────────────


def joint_action_distribution(
    action_history: dict[str, np.ndarray],
    n_actions_per_agent: list[int],
) -> np.ndarray:
    """Tensor empírico de joint actions: shape = action_shape.

    Útil para BoS/Chicken (heatmap de coordinación) y para detectar
    equilibrios correlacionados.
    """
    agents = list(action_history.keys())
    shape = tuple(n_actions_per_agent)
    dist = np.zeros(shape, dtype=float)
    T = len(action_history[agents[0]])
    for t in range(T):
        idx = tuple(int(action_history[a][t]) for a in agents)
        dist[idx] += 1
    return dist / T


# ─── Timing helpers ────────────────────────────────────────────────────────────


class Timer:
    """Context manager para medir tiempo de bloques de código."""

    def __init__(self, name: str = "block"):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
        print(f"  ⏱  {self.name}: {self.elapsed:.2f}s")
