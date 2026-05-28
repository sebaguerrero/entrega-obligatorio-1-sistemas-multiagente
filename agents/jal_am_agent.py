from dataclasses import dataclass
import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID


@dataclass
class JALAMAgentConfig:
    # Tasa de aprendizaje (alpha) usada en la regla de actualización de
    # Q-learning sobre acciones CONJUNTAS. A diferencia de IQL, acá Q se
    # indexa por (s, a_conjunta) — el espacio es mucho más grande, así
    # que conviene un alpha algo más bajo si los empates son frecuentes.
    alpha: float = 0.1

    # Factor de descuento (gamma) que pondera la importancia de las
    # recompensas futuras. El bootstrap usa el máximo de AV sobre la
    # propia acción en el siguiente estado, no max Q directo — esa es
    # la diferencia clave con IQL.
    gamma: float = 0.99

    # Valor mínimo al que puede decaer epsilon. Garantiza un piso de
    # exploración aún después de muchos pasos de entrenamiento.
    min_epsilon: float = 0.01

    # Horizonte para el decaimiento lineal de epsilon desde 1.0 hasta
    # `min_epsilon`. A partir de `max_t` pasos, epsilon se mantiene en
    # `min_epsilon`.
    max_t: int = 1000

    # Semilla para el generador de números aleatorios LOCAL del agente.
    seed: int | None = None

    # Valor inicial con el que se inicializan las entradas Q(s, a) para
    # acciones conjuntas no vistas. 0.0 es neutro y coincide con el
    # pseudocódigo de la Clase 7 slide 11.
    initial_q: float = 0.0


class JALAMAgent(Agent):
    """Joint Action Learning with Agent Modeling (Clase 7, slide 11).

    Para cada estado s mantiene:
      - Q(s, a_conjunta): tensor de forma `(|A_1|, ..., |A_n|)`. Cada
        entrada es el valor que recibe ESTE agente si se juega esa
        acción conjunta.
      - Modelo empírico π_j(· | s) de cada oponente j ≠ i, vía conteos
        normalizados con Laplace smoothing (arranca en 1.0 → prior
        uniforme, equivalente al "π_j = U(A_j)" del pseudocódigo).

    Valor esperado de cada acción propia bajo la creencia sobre los
    oponentes, asumiendo independencia entre ellos (factorización de
    π_-i, alineado con Albrecht & Stone Cap. 6.2):

        $AV_i(s, a_i) = \\sum_{a_{-i}} Q_i(s, (a_i, a_{-i})) \\cdot \\prod_{j \\neq i} \\pi_j(a_j | s)$

    Selección de acción: ε-greedy sobre AV. Target del Q-learning:
    $r + \\gamma \\max_{a_i'} AV_i(s', a_i')$ — bootstrap con el mejor
    AV en s' bajo la creencia actual de oponentes.

    Orden de updates (slide 11): primero se actualiza el modelo π_j de
    cada oponente con la acción observada, después se actualiza Q usando
    AV calculado con el π actualizado.
    """

    def __init__(
        self,
        game: SimultaneousGame,
        agent: AgentID,
        config: JALAMAgentConfig | None = None,
    ) -> None:
        super().__init__(game=game, agent=agent)

        self.config = config if config is not None else JALAMAgentConfig()

        # RNG LOCAL al agente, no contamina el estado global de NumPy.
        self._rng = np.random.default_rng(self.config.seed)

        # Lista ORDENADA de todos los agentes del juego. El orden importa
        # porque las acciones conjuntas se almacenan como tuplas indexadas
        # posicionalmente y los EJES del tensor Q siguen este orden.
        self.agents: list[AgentID] = list(self.game.agents)
        self.my_idx: int = self.agents.index(self.agent)

        # JAL-AM necesita conocer los espacios de acción de TODOS los
        # agentes (para indexar el tensor Q por acción conjunta).
        self.n_actions: dict[AgentID, int] = {a: self.game.num_actions(a) for a in self.agents}
        self.action_shape: tuple[int, ...] = tuple(self.n_actions[a] for a in self.agents)

        # Lista de acciones propias (puede no ser 0..n-1 si el juego usa
        # action_space.start ≠ 0). Se usa para muestrear exploración.
        self._own_actions: list[int] = list(self.game.action_iter(self.agent))

        # Tabla Q por estado (lazy).
        self.q: dict[tuple, np.ndarray] = {}

        # Modelo de cada oponente (lazy): conteos con Laplace smoothing.
        self.counts: dict[AgentID, dict[tuple, np.ndarray]] = {
            j: {} for j in self.agents if j != self.agent
        }

        # Tensores por defecto para LECTURA SIN MUTACIÓN — se usan
        # cuando se evalúa AV / Q en un estado nuevo (en `_action_value`
        # o `policy`) sin querer crear entradas en la tabla. Igual al
        # estado inicial de un estado recién creado.
        self._default_q: np.ndarray = np.full(self.action_shape, self.config.initial_q, dtype=float)
        self._default_counts: dict[AgentID, np.ndarray] = {
            j: np.ones(self.n_actions[j], dtype=float)
            for j in self.agents if j != self.agent
        }
        self._default_policies: dict[AgentID, np.ndarray] = {
            j: self._default_counts[j] / self._default_counts[j].sum()
            for j in self.counts
        }

        self.t: int = 0
        self.learn: bool = True

        # Estado en el que se eligió la última acción. La acción conjunta
        # NO se guarda acá: se la recupera en `update()` vía
        # `observe_action()` porque en `action()` sólo se conoce la propia.
        self.last_state: tuple | None = None

    def _state_key(self, obs) -> tuple | None:
        if obs is None:
            return None
        # En juegos one-shot puros el `observe()` puede devolver un dict
        # (ej. ThreePlayers devuelve el dict de acciones del último step).
        # En ese caso no hay un "estado" real en el sentido MDP, así que
        # usamos una clave fija — JAL-AM colapsa a un bandit contextual
        # con un solo estado.
        if isinstance(obs, dict):
            return ()
        # Las observaciones son arrays; tupla aplanada → hasheable.
        return tuple(np.asarray(obs).flatten().tolist())

    def _q_table(self, state: tuple) -> np.ndarray:
        # Acceso CON MUTACIÓN: si el estado no existe, se crea con
        # `initial_q`. Sólo usar desde `update()` cuando vamos a
        # escribir en la tabla.
        if state not in self.q:
            self.q[state] = np.full(self.action_shape, self.config.initial_q, dtype=float)
        return self.q[state]

    def _q_read(self, state: tuple) -> np.ndarray:
        # Acceso SIN MUTACIÓN para evaluación. Devuelve el tensor del
        # estado si existe, o el default si no — sin crear la entrada.
        # Imprescindible para `policy()` y para evaluar AV en `next_state`
        # durante el bootstrap, donde NO queremos inflar la tabla con
        # entradas que sólo se leyeron una vez.
        return self.q.get(state, self._default_q)

    def _opp_counts(self, j: AgentID, state: tuple) -> np.ndarray:
        # Acceso CON MUTACIÓN: si el estado no existe para el modelo de j,
        # se inicializa con Laplace smoothing (1.0 por acción → π_j
        # inicial uniforme, alineado al pseudocódigo "π_j(·|s) = U(A_j)").
        bucket = self.counts[j]
        if state not in bucket:
            bucket[state] = np.ones(self.n_actions[j], dtype=float)
        return bucket[state]

    def _opp_policy_read(self, j: AgentID, state: tuple) -> np.ndarray:
        # Acceso SIN MUTACIÓN: si el estado no existe, devuelve la
        # política uniforme cacheada (default). No crea entrada en
        # `self.counts[j]`. Usar para evaluación.
        if state in self.counts[j]:
            c = self.counts[j][state]
            return c / c.sum()
        return self._default_policies[j]

    def _epsilon(self) -> float:
        # Decaimiento lineal de 1.0 a min_epsilon a lo largo de max_t pasos.
        frac = min(1.0, self.t / max(1, self.config.max_t))
        return max(self.config.min_epsilon, 1.0 - frac * (1.0 - self.config.min_epsilon))

    def _action_value(self, state: tuple) -> np.ndarray:
        # AV_i(s, a_i) = Σ_{a_-i} Q(s, a) · Π_{j≠i} π_j(a_j|s)
        # Lectura SIN MUTACIÓN: usa `_q_read` y `_opp_policy_read` para
        # no crear entradas en la tabla cuando se evalúa un estado nuevo
        # (típico durante el bootstrap sobre `next_state`).
        #
        # Estrategia vectorizada: tensor `joint_prob` con la probabilidad
        # conjunta de los oponentes (tamaño 1 en la dim propia para
        # broadcastear sobre a_i), multiplicado element-wise con Q, y
        # sumado sobre los ejes ajenos al propio.
        q = self._q_read(state)
        prob_shape = list(self.action_shape)
        prob_shape[self.my_idx] = 1
        joint_prob = np.ones(prob_shape, dtype=float)
        for k, agent_k in enumerate(self.agents):
            if agent_k == self.agent:
                continue
            pi_k = self._opp_policy_read(agent_k, state)
            shape_k = [1] * len(self.agents)
            shape_k[k] = self.n_actions[agent_k]
            joint_prob = joint_prob * pi_k.reshape(shape_k)
        weighted = q * joint_prob
        sum_axes = tuple(k for k in range(len(self.agents)) if k != self.my_idx)
        return weighted.sum(axis=sum_axes)

    def reset(self) -> None:
        # Al inicio de cada episodio se borra el último estado para no
        # construir una transición espuria entre estados de episodios
        # distintos.
        self.last_state = None

    def update(self) -> None:
        if not self.learn:
            return
        if self.last_state is None:
            return

        # `observe_action()` devuelve la acción CONJUNTA del último step
        # como tupla en el orden de `self.agents`.
        joint_action = self.game.observe_action(self.agent)
        if joint_action is None:
            return
        ja = tuple(int(joint_action[k]) for k in range(len(self.agents)))

        # Orden alineado al pseudocódigo (slide 11):
        # 1. PRIMERO actualizar el modelo π_j de cada oponente con la
        #    acción que acabó de jugar.
        for j in self.counts:
            j_idx = self.agents.index(j)
            self._opp_counts(j, self.last_state)[ja[j_idx]] += 1

        # 2. DESPUÉS actualizar Q usando AV calculado con el π recién
        #    actualizado. El AV(s', ·) hace bootstrap sobre el modelo
        #    NUEVO de oponentes.
        r_i = self.game.reward(self.agent)
        next_state = self._state_key(self.game.observe(self.agent))
        row = self._q_table(self.last_state)
        q_sa = row[ja]
        if self.game.done():
            target = r_i
        else:
            target = r_i + self.config.gamma * np.max(self._action_value(next_state))
        row[ja] = q_sa + self.config.alpha * (target - q_sa)

        self.t += 1

    def action(self):
        state = self._state_key(self.game.observe(self.agent))

        if self.learn and self._rng.random() < self._epsilon():
            # Exploración: acción uniforme sobre las acciones propias.
            # Se usa `_own_actions` (de `action_iter`) en vez de
            # `integers(n_actions)` para ser robusto ante juegos con
            # `action_space.start ≠ 0`.
            a_i = int(self._rng.choice(self._own_actions))
        else:
            # Greedy con desempate aleatorio (evita el sesgo de argmax
            # hacia el primer índice cuando varias acciones empatan).
            av = self._action_value(state)
            best = np.flatnonzero(av == av.max())
            a_i = int(self._rng.choice(best))

        self.last_state = state
        return a_i

    def policy(self):
        # Política greedy sobre el estado actual SIN MUTACIÓN: si el
        # estado nunca se vio, devuelve la política uniforme implícita
        # del default (todas las acciones empatadas en `initial_q`).
        state = self._state_key(self.game.observe(self.agent))
        av = self._action_value(state)
        best = np.flatnonzero(av == av.max())
        learned_policy = np.zeros(self.n_actions[self.agent])
        learned_policy[best] = 1.0 / len(best)
        return learned_policy
