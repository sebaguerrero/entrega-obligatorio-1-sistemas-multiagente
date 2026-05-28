from itertools import product
from functools import reduce
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID


class FictitiousPlay(Agent):
    """Fictitious Play (Brown 1951).

    Cada agente mantiene un MODELO empírico de los OPONENTES (no de sí
    mismo — Clase 5 slide 3: "countᵗ_{p,q}(a_q) es la cantidad de veces
    que p observó a q ≠ p jugar a_q") contando cuántas veces los vio
    jugar cada acción. Normalizando esos conteos obtiene una estimación
    de la política de cada oponente, asume que seguirán jugando con esa
    distribución (hipótesis de **independencia entre oponentes**, slide 3
    footnote), y elige la "best response" — la acción que MAXIMIZA su
    utilidad esperada bajo esa creencia.

    En juegos zero-sum two-player con un equilibrio mixto único (RPS, MP),
    el promedio empírico de las propias acciones converge al equilibrio
    de Nash (Robinson 1951). En juegos coordinativos (BoS, Chicken) puede
    converger a uno de los equilibrios puros dependiendo de la
    inicialización de los conteos. Shapley (1964) mostró que puede no
    converger y exhibir ciclos en otros casos.

    Para reporte de la propia "política aprendida" se mantiene un
    contador separado `_own_count` con las frecuencias empíricas de las
    propias acciones (no usado por el algoritmo, sólo para `policy()`).
    """

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)

        # RNG LOCAL al agente (evita contaminar el estado global de NumPy
        # como hacía la inicialización original con `np.random.seed`). Si
        # se instancian múltiples agentes con seeds distintos, cada uno
        # mantiene su propio stream reproducible.
        self._rng = np.random.default_rng(seed)

        # Conteos de acciones de los OPONENTES (sólo j ≠ self.agent, como
        # exige el pseudocódigo de Clase 5 slide 3). Si se da `initial`
        # se respeta para cada oponente; si no, se inicializa con valores
        # aleatorios entre 1 y 9. La inicialización aleatoria rompe
        # simetrías en juegos coordinativos (BoS, Chicken) y evita
        # utilidades empatadas iniciales. dtype=int porque son CONTEOS.
        self.count: dict[AgentID, ndarray] = {
            a: np.array(initial[a], dtype=int) if initial is not None
               else self._rng.integers(1, 10, size=game.num_actions(a))
            for a in game.agents if a != self.agent
        }

        # Política aprendida de los oponentes: conteos normalizados a
        # probabilidad. Es la estimación π̂_q(·) bajo el supuesto de
        # que las observaciones son i.i.d.
        self.learned_policy: dict[AgentID, ndarray] = {
            a: self.count[a] / np.sum(self.count[a])
            for a in self.count
        }

        # Conteo separado de las acciones PROPIAS, sólo para reportar
        # `policy()`. NO se usa en el algoritmo (FP toma best response
        # contra los oponentes, no contra sí mismo). Arranca uniforme
        # en 1 para que la política reportada al inicio sea uniforme
        # y no haya división por cero.
        self._own_count: ndarray = np.ones(game.num_actions(self.agent), dtype=int)

        # Cache LAZY de la matriz de pagos. La matriz es estática en
        # juegos one-shot (no cambia entre rondas), así que recomputarla
        # cada `action()` es desperdicio. Se llena la primera vez que se
        # llama `get_rewards()` y se reutiliza. NOTA: si FP se aplicara
        # a un juego con estado, este cache sería incorrecto — pero FP
        # no aplica a juegos secuenciales (Foraging) de todas formas.
        self._rewards_cache: dict[tuple, float] | None = None

    def get_count(self):
        # Getter público para inspección desde notebooks (debugging,
        # gráficos de convergencia de los modelos de oponentes).
        return self.count

    def get_rewards(self) -> dict:
        # Si ya computamos la matriz de pagos, la devolvemos cacheada.
        # En juegos one-shot la matriz no cambia entre rondas, así que
        # este cache reduce drásticamente el costo de `action()`
        # (en Blotto con S=4, N=3 son 225 simulaciones por step → 1 sola
        # vez en lugar de 10k veces).
        if self._rewards_cache is not None:
            return self._rewards_cache

        # Primera vez: enumeramos todas las acciones conjuntas posibles y
        # simulamos cada una en un clon del juego para leer la recompensa
        # de ESTE agente.
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))

        def play(joint_action):
            g.step(dict(zip(g.agents, joint_action)))
            r = g.reward(self.agent)
            g.reset()
            return (joint_action, r)

        rewards: dict[tuple, float] = dict(map(play, product(*agents_actions)))
        self._rewards_cache = rewards
        return rewards

    def get_utility(self):
        # Utilidad esperada de cada acción propia bajo la creencia actual
        # sobre los oponentes (`learned_policy`):
        #   U(a_i) = Σ_{a_-i} r(a_i, a_-i) · Π_{j ≠ i} π_j(a_j)
        # Asume INDEPENDENCIA entre oponentes (slide 3 footnote).
        rewards = self.get_rewards()
        utility = np.zeros(self.game.num_actions(self.agent))
        my_index = self.game.agents.index(self.agent)

        for joint_action, reward in rewards.items():
            # Probabilidad conjunta de las acciones de los OTROS agentes.
            # `reduce` itera sobre (índice, agente) FILTRANDO al propio
            # y multiplicando π_j(a_j) para cada oponente j.
            prob = reduce(lambda p, ia: p * self.learned_policy[ia[1]][joint_action[ia[0]]],
                          filter(lambda ia: ia[1] != self.agent, enumerate(self.game.agents)), 1.0)
            # Acumular reward ponderado por la prob en la entrada de
            # `utility` correspondiente a la acción propia de esta
            # combinación.
            utility[joint_action[my_index]] += prob * reward
        return utility

    def bestresponse(self):
        # Best response = acción que maximiza la utilidad esperada bajo la
        # creencia actual de los oponentes. Se desempata aleatoriamente
        # entre todas las acciones que alcanzan el máximo, para evitar
        # el sesgo sistemático de `np.argmax` hacia el primer índice
        # (que en juegos con empates iniciales puede causar oscilaciones
        # espurias que no son del algoritmo sino del tiebreak).
        utility = self.get_utility()
        best = np.flatnonzero(utility == utility.max())
        return int(self._rng.choice(best))

    def update(self) -> None:
        # `game.observe(self.agent)` devuelve el diccionario de acciones
        # de la última ronda. Si todavía no se jugó ninguna ronda, es
        # None y no hay nada que actualizar.
        actions = self.game.observe(self.agent)
        if actions is None:
            return

        # Actualizar el conteo de cada OPONENTE con la acción que jugó
        # y recalcular su política empírica. NO modelamos al propio
        # agente (pseudocódigo Clase 5 slide 3).
        for agent in self.count:
            self.count[agent][actions[agent]] += 1
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

        # Actualizar el conteo propio sólo para reporte (no se usa en el
        # algoritmo). Esto permite que `policy()` muestre la frecuencia
        # empírica de las propias acciones, que en self-play converge
        # al equilibrio de Nash en juegos zero-sum.
        self._own_count[actions[self.agent]] += 1

    def action(self):
        # Antes de elegir, se incorpora la información de la ronda
        # anterior actualizando los conteos y la política aprendida.
        # Después se juega la best response a la política aprendida
        # actualizada.
        self.update()
        return self.bestresponse()

    def policy(self):
        # Frecuencia empírica de las propias acciones (no usada por el
        # algoritmo). En self-play converge al equilibrio de Nash en
        # juegos zero-sum con equilibrio mixto único (RPS, MP).
        return self._own_count / np.sum(self._own_count)
