"""Plot helpers compartidos por las notebooks del equipo.

Estas funciones reciben datos en los formatos estándar producidos por
`utils.experiments` (un `history` dict de `play_oneshot()` o un DataFrame
largo de `run_oneshot()`/`run_sequential()`) y devuelven una figura matplotlib.

Si la función recibe `save='juego/nombre'`, además guarda automáticamente la
figura en `<owner>/images/<juego>/<nombre>.png` vía `utils.storage._save_fig`.

Convención de formatos esperados:

  - `history`: dict producido por `play_oneshot()`, con claves
    `policy_history`, `action_history`, `reward_history`,
    `curr_policy_history` (sólo RM), `cum_regrets_history` (sólo RM).
    Cada valor es dict[agent → ndarray].

  - DataFrame: columnas `seed`, `iter` (o `episode`), `agent`, `metric`,
    `value`. Producido por `run_oneshot()` / `run_sequential()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.storage import save_fig as _save_fig


def _slice_history(history: dict, n_iters: int | None) -> dict:
    """Devuelve una copia del `history` recortada a las primeras `n_iters`
    iteraciones; si `n_iters is None`, devuelve el original sin tocar."""
    if n_iters is None:
        return history
    out = {}
    for k, v in history.items():
        if isinstance(v, dict):
            out[k] = {a: arr[:n_iters] for a, arr in v.items()}
        else:
            out[k] = v
    return out


def plot_policy_evolution(
    history: dict,
    action_labels: list[str] | None = None,
    nash: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple = (11, 4),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Plot estándar de evolución de políticas para juegos one-shot pequeños.

    Una columna por agente, una línea por acción. Si se da `nash`, dibuja
    líneas horizontales punteadas con el equilibrio teórico para comparar.

    Útil para juegos con pocas acciones (MP=2, RPS=3, BoS/Chicken=2,
    ThreePlayers=2). Para Blotto/Cournot (muchas acciones) ver
    `plot_policy_heatmap` y `plot_expected_value_evolution`.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    agents = list(history["policy_history"].keys())
    n_agents = len(agents)
    fig, axes = plt.subplots(1, n_agents, figsize=figsize, sharey=True)
    if n_agents == 1:
        axes = [axes]
    for i, agent in enumerate(agents):
        ax = axes[i]
        ph = history["policy_history"][agent]
        n_actions = ph.shape[1]
        labels = action_labels or [f"a{k}" for k in range(n_actions)]
        for k in range(n_actions):
            ax.plot(ph[:, k], label=labels[k], alpha=0.8)
        if nash is not None:
            for k in range(min(len(nash), n_actions)):
                ax.axhline(y=float(nash[k]), linestyle="--", color=f"C{k}", alpha=0.4)
            ax.plot([], [], linestyle="--", color="gray", alpha=0.6, label="Nash")
        ax.set_title(f"Política — {agent}")
        ax.set_xlabel("iteración")
        if i == 0:
            ax.set_ylabel("probabilidad")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_policy_heatmap(
    history: dict,
    agent: str | None = None,
    title: str | None = None,
    figsize: tuple = (10, 4),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Heatmap de la política aprendida a lo largo del tiempo.

    Útil para juegos con MUCHAS acciones (Blotto 15+, Cournot 21) donde
    `plot_policy_evolution` con una línea por acción se vuelve ilegible.

    Eje X: iteración. Eje Y: acción. Color: probabilidad.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    agents = [agent] if agent else list(history["policy_history"].keys())
    fig, axes = plt.subplots(1, len(agents), figsize=figsize, squeeze=False)
    for i, a in enumerate(agents):
        ax = axes[0, i]
        ph = history["policy_history"][a]  # (T, n_actions)
        im = ax.imshow(ph.T, aspect="auto", cmap="viridis", vmin=0, vmax=ph.max())
        fig.colorbar(im, ax=ax, label="p(a)")
        ax.set_title(f"Política aprendida — {a}")
        ax.set_xlabel("iteración")
        ax.set_ylabel("acción (índice)")
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_expected_value_evolution(
    history: dict,
    action_values: np.ndarray,
    nash_value: float | None = None,
    ylabel: str = "E[acción]",
    title: str | None = None,
    figsize: tuple = (10, 4),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Plot de E[acción] = Σ p(a) · value(a) por iteración, para cada agente.

    Útil cuando las acciones son índices de un valor continuo (Cournot:
    índice 0..20 mapea a q ∈ [0, 10]). Plotea la cantidad esperada bajo la
    política aprendida vs el Nash analítico, si se provee.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    fig, ax = plt.subplots(figsize=figsize)
    for a in history["policy_history"]:
        ph = history["policy_history"][a]  # (T, n_actions)
        # E[v] por iteración
        ev = (ph * action_values[np.newaxis, :]).sum(axis=1)
        ax.plot(ev, label=a, alpha=0.8)
    if nash_value is not None:
        ax.axhline(
            y=nash_value,
            linestyle="--",
            color="red",
            alpha=0.6,
            label=f"Nash analítico = {nash_value:.2f}",
        )
    ax.set_xlabel("iteración")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_distance_to_nash(
    history: dict,
    nash,
    ax=None,
    ylabel: str = "Distancia L2 al equilibrio",
    title: str | None = None,
    figsize: tuple = (10, 4),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Distancia euclídea entre la política aprendida y el Nash teórico vs iteración.

    Una línea por agente. Si RM/FP cumplen, la curva debe tender a 0 (o a un
    valor pequeño con oscilación residual en el caso de RM).

    Parameters
    ----------
    nash : ndarray (n_actions,) o dict[agent → ndarray]
        Si es un ndarray, se usa el mismo Nash para todos los agentes (típico
        en juegos simétricos como MP, RPS). Si es dict, se usa el Nash
        correspondiente a cada agente.
    ax : matplotlib Axes, optional
        Si se da, dibuja en ese ax (no crea figura ni guarda). Útil para
        componer en un layout de subplots.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=figsize)
    for a, ph in history["policy_history"].items():
        nash_a = nash[a] if isinstance(nash, dict) else nash
        nash_a = np.asarray(nash_a, dtype=float)
        dist = np.linalg.norm(ph[:, : len(nash_a)] - nash_a, axis=1)
        ax.plot(dist, label=a, alpha=0.8)
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("iteración")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    if created:
        ax.figure.tight_layout()
        _save_fig(ax.figure, save)
    return ax


def plot_distance_to_closest_nash(
    history: dict,
    nash_dicts: list,
    title: str | None = None,
    figsize: tuple = (10, 4),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Distancia L2 al Nash puro más cercano cuando hay múltiples Nash (config 3 de ThreePlayers).

    Para cada agente y cada paso, calcula la distancia a cada Nash candidato y reporta el mínimo.

    Parameters
    ----------
    nash_dicts : list[dict[agent → ndarray]]
        Lista de Nash one-hot. Cada elemento es un dict {agent: ndarray} con la acción
        del Nash correspondiente. La distancia reportada es el mínimo sobre todos los Nash.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    ph = history["policy_history"]
    agents = list(ph.keys())
    fig, ax = plt.subplots(figsize=figsize)
    for a in agents:
        pol = ph[a]
        n = pol.shape[1]
        dists = []
        for hots in nash_dicts:
            hot = np.asarray(hots[a][:n], dtype=float)
            dists.append(np.linalg.norm(pol - hot, axis=1))
        d = np.stack(dists, axis=1).min(axis=1)
        ax.plot(d, label=a, alpha=0.8)
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("iteración")
    ax.set_ylabel("Distancia L2 al Nash más cercano")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_cumulative_average_reward(
    history: dict,
    expected_value: float | None = None,
    ax=None,
    ylabel: str = "Reward promedio acumulado",
    title: str | None = None,
    figsize: tuple = (10, 4),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Reward promedio acumulado por iteración, una curva por agente.

    Computa $\\frac{1}{t}\\sum_{s \\le t} r_s$ por agente. Si se da
    `expected_value` (ej. 0 en juegos zero-sum como MP/RPS), dibuja una línea
    horizontal punteada para comparar el reward observado contra el valor
    esperado del equilibrio.

    Si se pasa `ax`, dibuja en ese ax (no crea figura ni guarda) para
    componer en un layout de subplots.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=figsize)
    for a, rh in history["reward_history"].items():
        T = rh.shape[0]
        iters = np.arange(1, T + 1)
        ax.plot(iters, np.cumsum(rh) / iters, label=a, alpha=0.8)
    if expected_value is not None:
        ax.axhline(
            expected_value,
            color="black",
            linestyle=":",
            alpha=0.7,
            label=f"valor esperado {expected_value:g}",
        )
    ax.set_xlabel("iteración")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    if created:
        ax.figure.tight_layout()
        _save_fig(ax.figure, save)
    return ax


def plot_cumulative_regret(
    history: dict,
    action_labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple = (11, 4),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Evolución de los regrets acumulados (`cum_regrets`) de cada agente RM.

    Una línea por acción y un subplot por agente. Pasa por cero porque los
    regrets se acumulan algebraicamente (con signo); el patrón típico es que
    oscila entre positivo y negativo según qué acción habría sido mejor en
    cada ronda. Si hay un sesgo persistente del oponente, alguna línea se va
    positiva y otra negativa.

    Sólo aplica a agentes RM (los que tienen atributo `cum_regrets`). Para
    juegos one-shot con RM, llamar después de `run_demo(...)` que devuelve
    `cum_regrets_history` poblado.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    cr = history.get("cum_regrets_history", {})
    agents = list(cr.keys())
    if not agents:
        print("(no hay agentes RM en este experimento — `cum_regrets_history` vacío)")
        return None
    fig, axes = plt.subplots(1, len(agents), figsize=figsize, squeeze=False)
    for i, a in enumerate(agents):
        ax = axes[0, i]
        crh = cr[a]
        n_actions = crh.shape[1]
        labels = action_labels or [f"a{k}" for k in range(n_actions)]
        for k in range(n_actions):
            ax.plot(crh[:, k], label=labels[k], alpha=0.8)
        ax.axhline(0, color="black", linestyle="-", alpha=0.3)
        ax.set_title(f"Regrets acumulados — {a}")
        ax.set_xlabel("iteración")
        if i == 0:
            ax.set_ylabel("$G^t$ (regret acumulado)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_average_positive_regret(
    history: dict,
    title: str | None = None,
    figsize: tuple = (10, 4),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Promedio positivo del regret normalizado por $t$ vs iteración.

    Métrica: $\\frac{1}{t} \\cdot \\overline{\\max(G^t, 0)}$ donde el promedio es
    sobre acciones y $t$ es la iteración. Si RM cumple su garantía de no-regret,
    esta cantidad debe tender a 0 (regret crece sublinealmente en $t$).

    Sirve para validar la garantía teórica de Hart & Mas-Colell.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    cr = history.get("cum_regrets_history", {})
    agents = list(cr.keys())
    if not agents:
        print("(no hay agentes RM en este experimento)")
        return None
    fig, ax = plt.subplots(figsize=figsize)
    for a in agents:
        crh = cr[a]
        T = crh.shape[0]
        iters = np.arange(1, T + 1)
        positive_avg = np.maximum(crh, 0).mean(axis=1)
        ax.plot(positive_avg / iters, label=a, alpha=0.8)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("iteración $t$")
    ax.set_ylabel("Max regret positivo promedio")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_instantaneous_vs_average_policy(
    history: dict,
    action_labels: list[str] | None = None,
    agent: str | None = None,
    title: str | None = None,
    figsize: tuple = (12, 3.5),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Comparación de `curr_policy` (instantánea, oscila) vs `learned_policy`
    (promedio temporal, converge) para agentes RM.

    Un subplot por acción. La línea naranja (curr) puede oscilar fuerte —
    incluso saltar entre 0 y 1 — mientras que la azul (learned) se estabiliza
    suavemente. Este plot resuelve la confusión típica de "RM no converge"
    cuando alguien sólo mira `curr_policy`.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    curr = history.get("curr_policy_history", {})
    learned = history.get("policy_history", {})
    if agent is not None and agent not in curr:
        print(
            f"(agente {agent!r} no tiene curr_policy — sólo aplica a agentes RM; "
            f"disponibles: {list(curr.keys()) or 'ninguno'})"
        )
        return None
    agents = [agent] if agent else list(curr.keys())
    if not agents:
        print("(no hay agentes con curr_policy en este experimento)")
        return None
    a = agents[0]  # mostramos sólo el primero por compacidad
    n_actions = curr[a].shape[1]
    labels = action_labels or [f"a{k}" for k in range(n_actions)]

    fig, axes = plt.subplots(1, n_actions, figsize=figsize, sharey=True)
    if n_actions == 1:
        axes = [axes]
    for k in range(n_actions):
        ax = axes[k]
        ax.plot(
            curr[a][:, k],
            label="curr (instantánea)",
            alpha=0.5,
            color="C1",
            linewidth=1,
        )
        ax.plot(learned[a][:, k], label="learned (promedio)", color="C0", linewidth=2)
        ax.set_title(f"{a} — {labels[k]}")
        ax.set_xlabel("iteración")
        if k == 0:
            ax.set_ylabel("probabilidad")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        if k == n_actions - 1:
            ax.legend(loc="best", fontsize=8)
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_simplex_pairs(
    policies_a0: np.ndarray,
    policies_a1: np.ndarray | None = None,
    nash: np.ndarray | None = None,
    action_labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple = (14, 4),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Trayectoria sobre el simplex 3D proyectada en 3 subplots de pares de acciones.

    Para RPS, los 3 pares son (R-P), (P-S), (R-S). Alternativa al
    `plot_simplex_2d` baricéntrico: descompone la visualización en 3
    coordenadas cartesianas, lo que algunos lectores encuentran más
    intuitivo que la proyección baricéntrica. Matemáticamente equivalentes.
    """
    import matplotlib.pyplot as plt

    if max_iter is not None:
        policies_a0 = policies_a0[:max_iter]
        if policies_a1 is not None:
            policies_a1 = policies_a1[:max_iter]
    labels = action_labels or ["R", "P", "S"]
    pairs = [(0, 1), (1, 2), (0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (i, j) in zip(axes, pairs):
        ax.plot(
            policies_a0[:, i],
            policies_a0[:, j],
            "-",
            alpha=0.5,
            label="agent_0",
            color="C0",
        )
        ax.scatter(policies_a0[-1, i], policies_a0[-1, j], color="C0", s=60, zorder=3)
        if policies_a1 is not None:
            ax.plot(
                policies_a1[:, i],
                policies_a1[:, j],
                "-",
                alpha=0.5,
                label="agent_1",
                color="C1",
            )
            ax.scatter(
                policies_a1[-1, i], policies_a1[-1, j], color="C1", s=60, zorder=3
            )
        if nash is not None:
            ax.scatter(
                nash[i], nash[j], marker="x", s=200, color="red", label="Nash", zorder=4
            )
        ax.set_xlabel(f"P({labels[i]})")
        ax.set_ylabel(f"P({labels[j]})")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_qtable_growth(
    df,
    ax=None,
    title: str | None = None,
    figsize: tuple = (10, 4),
    hue: str = "agent",
    save: str | None = None,
    max_iter: int | None = None,
):
    """Plot del tamaño de la Q-table (`q_size`) vs iteración, agregado sobre seeds.

    Captura la cobertura del espacio de estados durante el entrenamiento —
    métrica que la consigna pide explícitamente para Foraging.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sub = df[df["metric"] == "q_size"]
    if max_iter is not None:
        sub = sub[sub["iter"] < max_iter]
    if len(sub) == 0:
        ax.set_title("No q_size data")
        return ax
    sns.lineplot(data=sub, x="iter", y="value", hue=hue, errorbar=("ci", 95), ax=ax)
    ax.set_xlabel("iteración")
    ax.set_ylabel("|Q-table| (estados visitados)")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    _save_fig(ax.figure if ax is not None else None, save)
    return ax


def plot_foraging_rewards(
    history: dict,
    title: str | None = None,
    figsize: tuple = (13, 4.5),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Reward promedio por iteración y curva de aprendizaje promedio.

    Espera el `history` producido por las notebooks de Foraging, con claves
    `avg_rewards` y `mean_reward`. Genera una figura con dos paneles: reward
    por agente y promedio entre agentes.
    """
    import matplotlib.pyplot as plt

    n = len(history["mean_reward"])
    if max_iter is not None:
        n = min(n, max_iter)
    iters = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for agent, rewards in history["avg_rewards"].items():
        agent_type = history.get("agent_types", {}).get(agent, "")
        label = f"{agent} ({agent_type})" if agent_type else agent
        axes[0].plot(iters, rewards[:n], label=label)
    axes[0].set_title("Reward promedio por episodio")
    axes[0].set_xlabel("iteración")
    axes[0].set_ylabel("reward promedio")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        iters, history["mean_reward"][:n], color="black", label="promedio entre agentes"
    )
    axes[1].set_title("Curva de aprendizaje")
    axes[1].set_xlabel("iteración")
    axes[1].set_ylabel("reward promedio")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_foraging_policy_stability(
    history: dict,
    title: str | None = None,
    figsize: tuple = (10, 4.5),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Distancia L2 entre políticas greedy consecutivas durante Foraging."""
    import matplotlib.pyplot as plt

    n = len(history["mean_reward"])
    if max_iter is not None:
        n = min(n, max_iter)
    iters = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=figsize)
    for agent, values in history["policy_delta"].items():
        agent_type = history.get("agent_types", {}).get(agent, "")
        label = f"{agent} ({agent_type})" if agent_type else agent
        ax.plot(iters, values[:n], label=label)
    ax.set_xlabel("iteración")
    ax.set_ylabel("distancia L2 entre políticas consecutivas")
    ax.set_title(title or "Estabilidad de políticas aprendidas")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_foraging_states_qtable(
    history: dict,
    title: str | None = None,
    figsize: tuple = (13, 4.5),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Estados registrados y tamaño de Q-table por agente en Foraging."""
    import matplotlib.pyplot as plt

    n = len(history["mean_reward"])
    if max_iter is not None:
        n = min(n, max_iter)
    iters = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for agent, values in history["visited_states"].items():
        agent_type = history.get("agent_types", {}).get(agent, "")
        label = f"{agent} ({agent_type})" if agent_type else agent
        axes[0].plot(iters, values[:n], label=label)
    axes[0].set_title("Estados visitados/registrados")
    axes[0].set_xlabel("iteración")
    axes[0].set_ylabel("cantidad de estados")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for agent, values in history["q_table_size"].items():
        agent_type = history.get("agent_types", {}).get(agent, "")
        label = f"{agent} ({agent_type})" if agent_type else agent
        axes[1].plot(iters, values[:n], label=label)
    axes[1].set_title("Tamaño de Q-table")
    axes[1].set_xlabel("iteración")
    axes[1].set_ylabel("entradas en Q-table")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_foraging_coordination(
    history: dict,
    title: str | None = None,
    figsize: tuple = (13, 4.5),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Métricas simples de coordinación para Foraging."""
    import matplotlib.pyplot as plt

    n = len(history["mean_reward"])
    if max_iter is not None:
        n = min(n, max_iter)
    iters = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(
        iters, history["avg_pairwise_distance"][:n], label="distancia promedio"
    )
    axes[0].set_title("Distancia entre agentes")
    axes[0].set_xlabel("iteración")
    axes[0].set_ylabel("distancia Manhattan promedio")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        iters, history["joint_load_rate"][:n], color="tab:green", label="LOAD conjunto"
    )
    axes[1].set_title("LOAD conjunto")
    axes[1].set_xlabel("iteración")
    axes[1].set_ylabel("fracción de pasos con LOAD de 2+ agentes")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_metric_with_band(
    df: pd.DataFrame,
    metric: str,
    ax=None,
    title: str | None = None,
    xlabel: str = "iter",
    ylabel: str | None = None,
    hue: str = "agent",
    save: str | None = None,
    max_iter: int | None = None,
):
    """Plot de una métrica vs `iter` con banda de confianza (sobre seeds).

    Pinta una curva por valor de `hue` (default: `agent`). La banda viene del
    intervalo de confianza 95% calculado por seaborn sobre las seeds.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    sub = df[df["metric"] == metric]
    if max_iter is not None and "iter" in sub.columns:
        sub = sub[sub["iter"] < max_iter]
    if len(sub) == 0:
        ax.set_title(f"{title or metric}: NO DATA")
        return ax
    sns.lineplot(data=sub, x="iter", y="value", hue=hue, errorbar=("ci", 95), ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    _save_fig(ax.figure if ax is not None else None, save)
    return ax


def plot_simplex_2d(
    policies_a0: np.ndarray,
    policies_a1: np.ndarray | None = None,
    nash: np.ndarray | None = None,
    ax=None,
    title: str | None = None,
    save: str | None = None,
    max_iter: int | None = None,
):
    """Visualización de trayectoria de políticas en RPS (3 acciones) sobre el simplex 2D.

    Parameters
    ----------
    policies_a0 : ndarray (T, 3)
        Trayectoria de política del agente 0.
    policies_a1 : ndarray (T, 3), optional
        Trayectoria del agente 1 (se grafica en otro color).
    nash : ndarray (3,), optional
        Punto del Nash teórico (se marca con una cruz).
    """
    import matplotlib.pyplot as plt

    if max_iter is not None:
        policies_a0 = policies_a0[:max_iter]
        if policies_a1 is not None:
            policies_a1 = policies_a1[:max_iter]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # Proyectar simplex 3D al plano: (p0, p1, p2) → (x, y) usando coords baricéntricas
    def simplex_to_xy(p):
        x = (
            0.5
            * (2 * p[..., 1] + p[..., 2])
            / (p[..., 0] + p[..., 1] + p[..., 2] + 1e-12)
        )
        y = (np.sqrt(3) / 2) * p[..., 2] / (p[..., 0] + p[..., 1] + p[..., 2] + 1e-12)
        return x, y

    # Dibujar triángulo
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(corners[:, 0], corners[:, 1], "k-", alpha=0.3)
    ax.text(0, -0.05, "Rock", ha="center")
    ax.text(1, -0.05, "Paper", ha="center")
    ax.text(0.5, np.sqrt(3) / 2 + 0.03, "Scissors", ha="center")

    x0, y0 = simplex_to_xy(policies_a0)
    ax.plot(x0, y0, "-", alpha=0.4, label="agent_0", color="C0")
    ax.scatter(x0[-1], y0[-1], color="C0", s=50, zorder=3)

    if policies_a1 is not None:
        x1, y1 = simplex_to_xy(policies_a1)
        ax.plot(x1, y1, "-", alpha=0.4, label="agent_1", color="C1")
        ax.scatter(x1[-1], y1[-1], color="C1", s=50, zorder=3)

    if nash is not None:
        xn, yn = simplex_to_xy(np.asarray(nash))
        ax.scatter(xn, yn, marker="x", s=200, color="red", label="Nash", zorder=4)

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    ax.legend()
    _save_fig(ax.figure if ax is not None else None, save)
    return ax


def plot_cournot_best_response(
    game,
    history: dict | None = None,
    action_values: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple = (8, 6),
    save: str | None = None,
    max_iter: int | None = None,
    jitter: float = 0.05,
    alpha: float = 0.2,
):
    """Funciones de mejor respuesta para Cournot 2 jugadores.

    Útil como visualización didáctica de la estructura del juego antes de
    correr experimentos: muestra las dos curvas BR (lineales en $q_{-i}$) y
    marca el Nash interior simétrico en su intersección.

    Si se pasa `history`, además se superponen los pares (q₀, q₁) efectivamente
    jugados durante el entrenamiento, lo que permite verificar visualmente que
    los agentes convergen al cruce teórico de las BR (el Nash).

    Parameters
    ----------
    game : Cournot
        Instancia del juego. Se leen `a`, `b`, `c`, `max_quantity` directamente
        y se computa el Nash vía `game.get_nash_equilibrium()`.
    history : dict, optional
        `history` dict (de `play_oneshot` / `cached_run`). Si se pasa, agrega
        un scatter de los pares (q₀, q₁) jugados encima del gráfico de BR.
    action_values : ndarray, optional
        Valores reales asociados a los índices de acción (ej. `Cournot._quantities`).
        Necesario si se pasa `history`.
    """
    import matplotlib.pyplot as plt

    a, b, c, max_q = game.a, game.b, game.c, game.max_quantity
    q_star, _, _ = game.get_nash_equilibrium()

    q_range = np.linspace(0, max_q, 100)
    br_curve = np.maximum(0, (a - c - b * q_range) / (2 * b))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        q_range,
        br_curve,
        "b-",
        linewidth=2,
        label="BR Empresa 1 ($q_1 = q_1^{BR}(q_2)$)",
    )
    ax.plot(
        br_curve,
        q_range,
        "r-",
        linewidth=2,
        label="BR Empresa 2 ($q_2 = q_2^{BR}(q_1)$)",
    )

    # Superponer pares jugados si se da history
    if history is not None and action_values is not None:
        ah = history["action_history"]
        agents = list(ah.keys())[:2]
        xs = np.asarray(ah[agents[0]], dtype=int)
        ys = np.asarray(ah[agents[1]], dtype=int)
        if max_iter is not None:
            xs = xs[:max_iter]
            ys = ys[:max_iter]
        av = np.asarray(action_values, dtype=float)
        xs_v = av[xs]
        ys_v = av[ys]
        step = float(av[1] - av[0]) if len(av) > 1 else 1.0
        rng = np.random.default_rng(0)
        xs_v = xs_v + rng.uniform(
            -jitter * step / 2, jitter * step / 2, size=xs_v.shape
        )
        ys_v = ys_v + rng.uniform(
            -jitter * step / 2, jitter * step / 2, size=ys_v.shape
        )
        ax.scatter(
            xs_v,
            ys_v,
            alpha=alpha,
            s=12,
            color="gray",
            edgecolor="none",
            label="pares (q₀, q₁) jugados",
        )

    ax.plot(
        q_star, q_star, "go", markersize=12, label=f"Nash ({q_star:.2f}, {q_star:.2f})"
    )
    ax.set_xlabel("$q_2$ (cantidad empresa 2)")
    ax.set_ylabel("$q_1$ (cantidad empresa 1)")
    ax.set_title(title or "Funciones de mejor respuesta — Cournot 2 jugadores")
    ax.set_xlim(0, max_q)
    ax.set_ylim(0, max_q)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_action_over_time(
    history: dict,
    action_values: np.ndarray | None = None,
    nash_value: float | None = None,
    title: str | None = None,
    figsize: tuple = (10, 4),
    save: str | None = None,
    max_iter: int | None = None,
    sample_rate: int = 1,
):
    """Cantidad efectivamente jugada por cada agente vs iteración (acción instantánea).

    Análogo al "Quantity Convergence" del notebook original de Cournot: muestra
    una serie temporal por agente con la cantidad que jugó en cada paso
    (en unidades reales si se pasa `action_values`), más una línea horizontal
    del Nash teórico. Útil para visualizar la convergencia de la acción
    muestreada (que oscila al inicio y se estabiliza) — complementaria a
    `plot_expected_value_evolution` que muestra la esperanza bajo la política.

    Parameters
    ----------
    sample_rate : int
        Si >1, hace downsample (`actions[::sample_rate]`) para reducir el
        ruido visual cuando hay muchas iteraciones.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    ah = history["action_history"]
    fig, ax = plt.subplots(figsize=figsize)
    for a, actions in ah.items():
        if action_values is not None:
            ys = np.asarray(action_values)[np.asarray(actions, dtype=int)]
        else:
            ys = np.asarray(actions)
        ts = np.arange(len(ys))
        if sample_rate > 1:
            ys = ys[::sample_rate]
            ts = ts[::sample_rate]
        ax.plot(ts, ys, alpha=0.55, linewidth=1, label=a)
    if nash_value is not None:
        ax.axhline(
            float(nash_value),
            linestyle="--",
            color="green",
            linewidth=2,
            label=f"Nash = {float(nash_value):g}",
        )
    ax.set_xlabel("iteración")
    ax.set_ylabel(
        "acción jugada (valor)"
        if action_values is not None
        else "acción jugada (índice)"
    )
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_reward_over_time(
    history: dict,
    expected_value: float | None = None,
    title: str | None = None,
    figsize: tuple = (10, 4),
    save: str | None = None,
    max_iter: int | None = None,
    sample_rate: int = 1,
):
    """Reward INSTANTÁNEO por agente vs iteración (no promedio acumulado).

    Análogo al "Profit Convergence" del notebook original de Cournot: serie
    temporal del reward de cada paso, con línea horizontal del valor teórico
    en equilibrio. Distinto de `plot_cumulative_average_reward` (que suaviza
    promediando hasta `t`); útil para ver la varianza del reward instantáneo.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    rh = history["reward_history"]
    fig, ax = plt.subplots(figsize=figsize)
    for a, rewards in rh.items():
        ys = np.asarray(rewards)
        ts = np.arange(len(ys))
        if sample_rate > 1:
            ys = ys[::sample_rate]
            ts = ts[::sample_rate]
        ax.plot(ts, ys, alpha=0.55, linewidth=1, label=a)
    if expected_value is not None:
        ax.axhline(
            float(expected_value),
            linestyle="--",
            color="green",
            linewidth=2,
            label=f"reward Nash = {float(expected_value):g}",
        )
    ax.set_xlabel("iteración")
    ax.set_ylabel("reward instantáneo")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_policy_bars(
    history: dict,
    action_values: np.ndarray | None = None,
    action_labels: list[str] | None = None,
    nash_value: float | None = None,
    title: str | None = None,
    figsize: tuple = (10, 4),
    save: str | None = None,
):
    """Bar plot de la POLÍTICA APRENDIDA final (`policy_history[a][-1]`), una serie por agente.

    Análogo al "Strategy Distribution" del notebook original de Cournot:
    barras superpuestas o agrupadas con la probabilidad final de cada
    acción para cada agente, con línea vertical en el Nash. Distinto de
    `plot_action_histogram` (que cuenta frecuencias empíricas de acciones
    JUGADAS, sin promediar la política) — éste muestra la política
    aprendida final (`policy()[-1]`).
    """
    import matplotlib.pyplot as plt

    ph = history["policy_history"]
    agents = list(ph.keys())
    n_agents = len(agents)
    n_actions = ph[agents[0]].shape[1]
    fig, ax = plt.subplots(figsize=figsize)
    if action_values is not None:
        xs = np.asarray(action_values[:n_actions], dtype=float)
        step = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
        bar_w = step * 0.85 / n_agents
        for i, a in enumerate(agents):
            offset = (i - (n_agents - 1) / 2) * bar_w
            ax.bar(
                xs + offset,
                ph[a][-1, :n_actions],
                width=bar_w,
                alpha=0.75,
                edgecolor="black",
                linewidth=0.4,
                label=a,
            )
        ax.set_xlabel("acción (valor)")
    else:
        xs = np.arange(n_actions)
        bar_w = 0.85 / n_agents
        for i, a in enumerate(agents):
            offset = (i - (n_agents - 1) / 2) * bar_w
            ax.bar(
                xs + offset,
                ph[a][-1, :n_actions],
                width=bar_w,
                alpha=0.75,
                edgecolor="black",
                linewidth=0.4,
                label=a,
            )
        if action_labels is not None:
            ax.set_xticks(xs)
            ax.set_xticklabels(list(action_labels)[:n_actions])
        ax.set_xlabel("acción (índice)")
    if nash_value is not None:
        ax.axvline(
            float(nash_value),
            linestyle="--",
            color="green",
            linewidth=2,
            label=f"Nash = {float(nash_value):g}",
        )
    ax.set_ylabel("probabilidad (política final)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_expected_value_instantaneous_vs_average(
    history: dict,
    action_values: np.ndarray,
    agent: str | None = None,
    nash_value: float | None = None,
    title: str | None = None,
    figsize: tuple = (10, 4),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Compara $E[acción]_t$ bajo la política INSTANTÁNEA vs PROMEDIO para agentes RM.

    Variante específica para Cournot/Blotto (juegos con muchas acciones y un
    valor continuo asociado a cada acción): en lugar de mostrar 21 subplots
    (uno por acción), reduce a 1 sola figura con dos líneas — `curr_policy`
    contra `learned_policy` — agregadas como esperanza sobre `action_values`.

    Sirve para visualizar el fenómeno típico de RM: la política instantánea
    oscila (E[q] salta entre valores), pero el promedio converge suavemente.
    Si se pasa `nash_value`, se dibuja la línea horizontal de referencia.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    curr = history.get("curr_policy_history", {})
    learned = history.get("policy_history", {})
    if agent is not None and agent not in curr:
        print(
            f"(agente {agent!r} no tiene curr_policy — sólo aplica a agentes RM; "
            f"disponibles: {list(curr.keys()) or 'ninguno'})"
        )
        return None
    agents = [agent] if agent else list(curr.keys())
    if not agents:
        print("(no hay agentes con curr_policy en este experimento)")
        return None
    a = agents[0]
    av = np.asarray(action_values, dtype=float)
    n_used = min(curr[a].shape[1], len(av))
    e_curr = (curr[a][:, :n_used] * av[np.newaxis, :n_used]).sum(axis=1)
    e_learned = (learned[a][:, :n_used] * av[np.newaxis, :n_used]).sum(axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(e_curr, label="curr (instantánea)", alpha=0.5, color="C1", linewidth=1)
    ax.plot(e_learned, label="learned (promedio)", color="C0", linewidth=2)
    if nash_value is not None:
        ax.axhline(
            float(nash_value),
            linestyle="--",
            color="red",
            alpha=0.6,
            label=f"Nash = {float(nash_value):g}",
        )
    ax.set_xlabel("iteración")
    ax.set_ylabel("E[acción]")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_action_histogram(
    history: dict,
    action_values: np.ndarray | None = None,
    action_labels: list[str] | None = None,
    nash_value: float | np.ndarray | None = None,
    agent: str | None = None,
    title: str | None = None,
    figsize: tuple = (11, 4),
    save: str | None = None,
    max_iter: int | None = None,
):
    """Histograma de las acciones efectivamente jugadas, una columna por agente.

    Lee `history["action_history"][a]` (ndarray (T,) de índices) y cuenta la
    frecuencia empírica de cada índice. Útil para Cournot/Blotto (ver
    concentración alrededor del Nash) y para ThreePlayers (ver concentración
    en la BR puro).

    Parameters
    ----------
    action_values : ndarray, optional
        Si se pasa (ej. `Cournot._quantities` ∈ [0, 10]), el eje X se grafica
        en unidades reales en lugar de índices. La barra para el índice `i`
        se ubica en `action_values[i]` con ancho `(action_values[1] - action_values[0]) * 0.9`.
    action_labels : list[str], optional
        Etiquetas categóricas (ej. `['T','B']` para ThreePlayers). Se usan
        como xticks. Ignorado si se pasa `action_values`.
    nash_value : float o ndarray, optional
        Línea(s) vertical(es) punteada(s) marcando el equilibrio teórico.
    agent : str, optional
        Si se pasa, sólo se grafica ese agente; si no, una columna por agente.
    max_iter : int, optional
        Trunca el `action_history` a las primeras `max_iter` rondas.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    ah = history["action_history"]
    agents = [agent] if agent else list(ah.keys())
    n_agents = len(agents)
    fig, axes = plt.subplots(1, n_agents, figsize=figsize, squeeze=False)
    for i, a in enumerate(agents):
        ax = axes[0, i]
        actions = np.asarray(ah[a])
        # policy_history.shape[1] carga el número real de acciones por agente;
        # actions.max()+1 lo undercount cuando no se jugaron todas.
        ph = history.get("policy_history", {}).get(a)
        if ph is not None and hasattr(ph, "shape") and ph.ndim >= 2 and ph.shape[1] > 0:
            n_actions = int(ph.shape[1])
        else:
            n_actions = int(actions.max()) + 1 if len(actions) > 0 else 0
        if action_values is not None:
            n_actions = max(n_actions, len(action_values))
        counts = np.bincount(actions, minlength=n_actions).astype(float)
        freq = counts / counts.sum() if counts.sum() > 0 else counts
        if action_values is not None:
            xs = np.asarray(action_values[:n_actions], dtype=float)
            width = float(xs[1] - xs[0]) * 0.9 if len(xs) > 1 else 1.0
            ax.bar(xs, freq, width=width, alpha=0.75, edgecolor="black", linewidth=0.4)
            ax.set_xlabel("acción (valor)")
        else:
            xs = np.arange(n_actions)
            ax.bar(xs, freq, alpha=0.75, edgecolor="black", linewidth=0.4)
            ax.set_xticks(xs)
            if action_labels is not None:
                ax.set_xticklabels(list(action_labels)[:n_actions])
            if n_actions > 0:
                ax.set_xlim(-0.5, n_actions - 0.5)
            ax.set_xlabel("acción (índice)")
        if nash_value is not None:
            nash_arr = np.atleast_1d(nash_value)
            for nv in nash_arr:
                ax.axvline(
                    float(nv),
                    linestyle="--",
                    color="red",
                    alpha=0.7,
                    label=f"Nash = {float(nv):g}" if nv == nash_arr[0] else None,
                )
            ax.legend(loc="best", fontsize=8)
        ax.set_ylabel("frecuencia empírica")
        ax.set_title(f"Acciones jugadas — {a}")
        ax.grid(alpha=0.3, axis="y")
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_actions_scatter(
    history: dict,
    action_values: np.ndarray | None = None,
    agents: list[str] | None = None,
    nash: tuple[float, float] | None = None,
    title: str | None = None,
    figsize: tuple = (6, 5),
    save: str | None = None,
    max_iter: int | None = None,
    jitter: float = 0.15,
    alpha: float = 0.25,
):
    """Scatter plot 2D del par de acciones (agent_i, agent_j) a lo largo del tiempo.

    Útil para Cournot 2 jugadores (nube `(q_0, q_1)` y cruz en `(q*, q*)`).
    Para juegos con 3+ jugadores pasar `agents=['agent_0', 'agent_2']` para
    elegir explícitamente el par.

    Parameters
    ----------
    action_values : ndarray, optional
        Mapeo índice → valor real (ej. `Cournot._quantities`). Si se pasa,
        los ejes se grafican en unidades reales.
    agents : list[str], optional
        Par de agentes a comparar. Por defecto, los primeros dos del history.
    nash : tuple, optional
        Punto Nash a marcar con cruz roja, en las mismas unidades que el eje
        (índices si `action_values is None`, valores reales en otro caso).
    jitter : float
        Magnitud de ruido aleatorio agregado a los puntos para que las
        coincidencias densas no se reduzcan a un único pixel.
    alpha : float
        Transparencia de los puntos para visualizar densidad.
    """
    import matplotlib.pyplot as plt

    history = _slice_history(history, max_iter)
    ah = history["action_history"]
    if agents is None:
        agents = list(ah.keys())[:2]
    if len(agents) != 2:
        raise ValueError(
            f"plot_actions_scatter espera 2 agentes, recibió {len(agents)}"
        )
    a_i, a_j = agents
    xs = np.asarray(ah[a_i], dtype=float)
    ys = np.asarray(ah[a_j], dtype=float)
    if action_values is not None:
        action_values = np.asarray(action_values, dtype=float)
        step = (
            float(action_values[1] - action_values[0])
            if len(action_values) > 1
            else 1.0
        )
        xs = action_values[xs.astype(int)]
        ys = action_values[ys.astype(int)]
        jitter_eff = jitter * step
    else:
        jitter_eff = jitter
    rng = np.random.default_rng(0)
    xs = xs + rng.uniform(-jitter_eff / 2, jitter_eff / 2, size=xs.shape)
    ys = ys + rng.uniform(-jitter_eff / 2, jitter_eff / 2, size=ys.shape)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xs, ys, alpha=alpha, s=14, edgecolor="none")
    if nash is not None:
        ax.scatter(
            [nash[0]],
            [nash[1]],
            marker="x",
            s=200,
            color="red",
            linewidths=2.5,
            zorder=5,
            label=f"Nash = ({nash[0]:g}, {nash[1]:g})",
        )
        ax.legend(loc="best", fontsize=9)
    if action_values is not None:
        ax.set_xlabel(f"{a_i} — acción (valor)")
        ax.set_ylabel(f"{a_j} — acción (valor)")
    else:
        ax.set_xlabel(f"{a_i} — acción (índice)")
        ax.set_ylabel(f"{a_j} — acción (índice)")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save)
    return fig


def plot_utility_space(
    equilibria: list[dict] | None = None,
    segments: list[dict] | None = None,
    trajectory: tuple | None = None,
    cumulative: bool = True,
    scatter: dict | np.ndarray | None = None,
    scatter_color: str | None = None,
    annotate_equilibria: bool = False,
    ax=None,
    title: str | None = None,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    xlabel: str = "utilidad agent_0",
    ylabel: str = "utilidad agent_1",
    cmap: str = "viridis",
    show_legend: bool = True,
    save: str | None = None,
):
    """Plot 2D del espacio de utilidades para juegos de 2 agentes.

    Eje X = utilidad de agent_0, eje Y = utilidad de agent_1. Permite marcar
    equilibrios teóricos (NE puros, NE mixto, CE fair, etc.), segmentos de
    referencia (ej. conjunto de equilibrios correlacionados), y superponer
    una trayectoria de utilidad acumulada (single-seed) y/o un scatter de
    utilidades finales por seed (multi-seed).

    Parámetros:
      equilibria: list[dict] — puntos a marcar. Cada dict acepta:
        'pos' (tupla X,Y, requerido), 'label', 'color' (default 'black'),
        'marker' (default 'o'), 's' (default 120), 'coord_text' (anotación
        custom; si no se da, se autoformatea desde pos), 'text_offset'
        (tupla en points; default (8, 8)).
      annotate_equilibria: si True, anota cada equilibrio con su coordenada
        al lado (usa 'coord_text' o autoformatea desde 'pos').
      segments: list[dict] — líneas dashed entre dos puntos. Cada dict:
        'start' (X,Y), 'end' (X,Y), 'label' (opcional).
      trajectory: tupla (rewards_0, rewards_1) de arrays de igual longitud.
        Si cumulative=True, se aplica cumsum/n antes de plotear (default).
        Se pinta con gradient de color por iteración.
      scatter: dict {nombre: ndarray(n, 2)} o ndarray(n, 2). Cada array es
        una nube de puntos finales en utilidad para una pairing.
      cumulative: aplica promedio acumulado a la trayectoria (default True).
      xlim, ylim: opcional. Si no se pasan, matplotlib auto-escala.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    if equilibria:
        for eq in equilibria:
            x, y = eq["pos"]
            ax.scatter(
                x,
                y,
                color=eq.get("color", "black"),
                s=eq.get("s", 120),
                marker=eq.get("marker", "o"),
                zorder=5,
                label=eq.get("label"),
            )
            if annotate_equilibria:
                text = eq.get("coord_text") or f"({x:g}, {y:g})"
                offset = eq.get("text_offset", (8, 8))
                ax.annotate(
                    text,
                    xy=(x, y),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=10,
                    color=eq.get("color", "black"),
                    ha="left",
                    va="bottom",
                    zorder=6,
                )

    if segments:
        for seg in segments:
            (x0, y0) = seg["start"]
            (x1, y1) = seg["end"]
            ax.plot(
                [x0, x1],
                [y0, y1],
                "k--",
                alpha=0.5,
                linewidth=1.5,
                label=seg.get("label"),
            )

    if trajectory is not None:
        r0 = np.asarray(trajectory[0], dtype=float)
        r1 = np.asarray(trajectory[1], dtype=float)
        if cumulative:
            r0 = np.cumsum(r0) / np.arange(1, len(r0) + 1)
            r1 = np.cumsum(r1) / np.arange(1, len(r1) + 1)
        pts = np.column_stack([r0, r1]).reshape(-1, 1, 2)
        segs_lc = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs_lc, cmap=cmap, linewidth=2, alpha=0.85)
        lc.set_array(np.arange(len(r0)))
        ax.add_collection(lc)
        ax.scatter(
            r0[-1],
            r1[-1],
            color="blue",
            s=180,
            marker="*",
            zorder=6,
            edgecolor="white",
            linewidth=1.5,
            label="final",
        )

    if scatter is not None:
        if isinstance(scatter, dict):
            for name, pts in scatter.items():
                pts = np.asarray(pts)
                kw = dict(
                    alpha=0.55,
                    s=70,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=4,
                    label=name,
                )
                if scatter_color is not None:
                    kw["color"] = scatter_color
                ax.scatter(pts[:, 0], pts[:, 1], **kw)
        else:
            pts = np.asarray(scatter)
            kw = dict(alpha=0.55, s=70, edgecolor="white", linewidth=0.5, zorder=4)
            if scatter_color is not None:
                kw["color"] = scatter_color
            ax.scatter(pts[:, 0], pts[:, 1], **kw)

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    if show_legend and ax.get_legend_handles_labels()[1]:
        ax.legend(loc="lower left", fontsize=8)

    _save_fig(ax.figure if ax is not None else None, save)
    return ax


def plot_joint_action_heatmap(
    dist: np.ndarray,
    action_labels: list[list[str]] | None = None,
    ax=None,
    title: str | None = None,
    save: str | None = None,
):
    """Heatmap de la distribución empírica de joint actions (2D, para 2 jugadores)."""
    import matplotlib.pyplot as plt

    if dist.ndim != 2:
        raise ValueError(
            f"plot_joint_action_heatmap espera dist 2D, recibió {dist.ndim}D"
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(dist, cmap="viridis", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, label="frecuencia empírica")
    if action_labels is not None and len(action_labels) == 2:
        ax.set_xticks(range(dist.shape[1]))
        ax.set_xticklabels(action_labels[1])
        ax.set_yticks(range(dist.shape[0]))
        ax.set_yticklabels(action_labels[0])
    ax.set_xlabel("agent_1")
    ax.set_ylabel("agent_0")
    if title:
        ax.set_title(title)
    # Anotar las celdas
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            ax.text(
                j,
                i,
                f"{dist[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if dist[i, j] < dist.max() * 0.6 else "black",
                fontsize=10,
            )
    _save_fig(ax.figure if ax is not None else None, save)
    return ax
