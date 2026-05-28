from dataclasses import dataclass
import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID


@dataclass
class IQLAgentConfig:
    # Tasa de aprendizaje (alpha) usada en la regla de actualización de
    # Q-learning. Cuanto más alta, más peso se le da a la nueva
    # información frente a la estimación previa de Q(s, a).
    alpha: float = 0.1

    # Factor de descuento (gamma) que pondera la importancia de las
    # recompensas futuras frente a la recompensa inmediata. Un gamma
    # cercano a 1 prioriza retornos a largo plazo (necesario en Foraging
    # porque la recompensa llega varios pasos después de moverse hacia
    # la fruta).
    gamma: float = 0.99

    # Valor mínimo al que puede decaer epsilon. Garantiza un piso de
    # exploración aún después de muchos pasos de entrenamiento.
    min_epsilon: float = 0.01

    # Horizonte del decaimiento lineal de epsilon desde 1.0 hasta
    # `min_epsilon`.
    max_t: int = 1000

    # Semilla para el RNG LOCAL del agente.
    seed: int | None = None

    # Q inicial para estados nuevos. 0.0 es neutro y coincide con el
    # pseudocódigo de Clase 7 slide 8.
    initial_q: float = 0.0


class IQLAgent(Agent):
    """Independent Q-Learning (Clase 7, slide 8).

    Cada agente trata al resto como parte del entorno: aprende su propia
    tabla Q(s, a_i) ignorando que la dinámica del entorno depende
    también de las acciones de los demás. Simple y escalable, pero no
    garantiza convergencia a Nash en juegos no estacionarios donde la
    política de los oponentes cambia durante el aprendizaje.

    Pseudocódigo (slide 8):

        Inicializar Q_i(s, a_i) = 0
        Repetir episodios:
            s ~ μ
            Repetir hasta s terminal:
                a_i = ε-greedy sobre Q_i(s, ·)
                Observar r_i, s'
                Q_i(s, a_i) ← Q_i(s, a_i) + α [r_i + γ máx_{a'} Q_i(s', a') − Q_i(s, a_i)]
                s ← s'
    """

    def __init__(
        self,
        game: SimultaneousGame,
        agent: AgentID,
        config: IQLAgentConfig | None = None,
    ) -> None:
        super().__init__(game=game, agent=agent)

        self.config = config if config is not None else IQLAgentConfig()

        # RNG LOCAL al agente, no contamina el estado global de NumPy.
        self._rng = np.random.default_rng(self.config.seed)

        # IQL sólo necesita conocer su propio espacio de acciones.
        self.n_actions: int = self.game.num_actions(self.agent)

        # Lista de acciones disponibles (puede no ser 0..n-1 si el juego
        # usa action_space.start ≠ 0). Se usa para muestrear exploración.
        self._actions: list[int] = list(self.game.action_iter(self.agent))

        # Tabla Q lazy: dict[state → ndarray].
        self.q: dict[tuple, np.ndarray] = {}

        # Vector Q por defecto para LECTURA SIN MUTACIÓN — se usa cuando
        # se quiere consultar el Q de un estado sin crearlo en la tabla
        # (en `policy()` y en el bootstrap sobre `next_state`).
        self._default_q: np.ndarray = np.full(self.n_actions, self.config.initial_q, dtype=float)

        self.t: int = 0
        self.learn: bool = True
        self.last_state: tuple | None = None
        self.last_action: int | None = None

    def _state_key(self, obs) -> tuple | None:
        if obs is None:
            return None
        # En juegos one-shot puros el `observe()` puede devolver un dict
        # (ej. ThreePlayers devuelve el dict de acciones del último step).
        # En ese caso no hay un "estado" real en el sentido MDP, así que
        # usamos una clave fija — IQL colapsa a un bandit con un único
        # estado.
        if isinstance(obs, dict):
            return ()
        # Tupla aplanada para usar la observación (array NumPy) como
        # clave hasheable de diccionario.
        return tuple(np.asarray(obs).flatten().tolist())

    def _q_row(self, state: tuple) -> np.ndarray:
        # Acceso CON MUTACIÓN: crea entrada con `initial_q` si el estado
        # no existe. Sólo usar desde `update()` cuando vamos a escribir.
        if state not in self.q:
            self.q[state] = np.full(self.n_actions, self.config.initial_q, dtype=float)
        return self.q[state]

    def _q_read(self, state: tuple) -> np.ndarray:
        # Acceso SIN MUTACIÓN: devuelve el vector del estado si existe, o
        # el default si no — sin inflar la tabla. Imprescindible para
        # `policy()` (inspección desde notebooks) y para el bootstrap
        # sobre `next_state` (que típicamente nunca se visita después).
        return self.q.get(state, self._default_q)

    def _epsilon(self) -> float:
        # Decaimiento lineal de 1.0 a min_epsilon a lo largo de max_t pasos.
        frac = min(1.0, self.t / max(1, self.config.max_t))
        return max(self.config.min_epsilon, 1.0 - frac * (1.0 - self.config.min_epsilon))

    def reset(self) -> None:
        # Al inicio de cada episodio se borra la última transición para
        # no construir una entre estados de episodios distintos.
        self.last_state = None
        self.last_action = None

    def update(self) -> None:
        if not self.learn:
            return
        if self.last_state is None or self.last_action is None:
            return

        # Contrato temporal: `update()` se llama DESPUÉS de
        # `game.step(actions)` en el loop del notebook, así que
        # `game.reward(self.agent)` es la recompensa de `last_action`
        # y `game.observe(self.agent)` es el siguiente estado s'.
        reward = self.game.reward(self.agent)
        next_state = self._state_key(self.game.observe(self.agent))

        # Row CON MUTACIÓN: vamos a escribir en él, así que sí queremos
        # crear la entrada si no existe.
        row = self._q_row(self.last_state)
        q_sa = row[self.last_action]

        # Target del Q-learning. Si el episodio terminó, no hay
        # recompensa futura. Si no, bootstrap con max Q(s', ·) usando
        # LECTURA SIN MUTACIÓN — no queremos crear entrada para s' si
        # no la habíamos visto antes.
        if self.game.done():
            target = reward
        else:
            target = reward + self.config.gamma * np.max(self._q_read(next_state))

        # Q(s, a) ← Q(s, a) + α (target − Q(s, a))
        row[self.last_action] = q_sa + self.config.alpha * (target - q_sa)
        self.t += 1

    def action(self):
        state = self._state_key(self.game.observe(self.agent))

        if self.learn and self._rng.random() < self._epsilon():
            # Exploración: acción uniforme sobre el conjunto real de
            # acciones disponibles (robusto a `action_space.start ≠ 0`).
            a = int(self._rng.choice(self._actions))
        else:
            # Greedy con desempate aleatorio. Usamos `_q_read` para
            # LEER sin crear entrada — pero después en `update()` el
            # estado SÍ se va a crear vía `_q_row(self.last_state)`.
            q_values = self._q_read(state)
            best = np.flatnonzero(q_values == q_values.max())
            a = int(self._rng.choice(best))

        self.last_state = state
        self.last_action = a
        return a

    def policy(self):
        # Política greedy sobre el estado actual SIN MUTACIÓN: si el
        # estado nunca se vio, devuelve uniforme (porque el default
        # tiene todos los Q iguales a `initial_q`).
        state = self._state_key(self.game.observe(self.agent))
        q_values = self._q_read(state)
        learned_policy = np.zeros(self.n_actions)
        best = np.flatnonzero(q_values == q_values.max())
        learned_policy[best] = 1.0 / len(best)
        return learned_policy
