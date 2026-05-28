import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict


class RegretMatching(Agent):
    """Regret Matching (Hart & Mas-Colell 2000; Clase 6 slide 5).

    En cada ronda calcula el "regret" contrafáctico de cada acción
    alternativa (cuánto mejor habría sido jugarla en vez de la que
    efectivamente se jugó, manteniendo fijas las acciones del resto),
    acumula esos regrets a lo largo del tiempo, y juega proporcionalmente
    a los regrets POSITIVOS acumulados.

    El promedio temporal de las políticas (`learned_policy`) converge a
    un equilibrio correlacionado en juegos one-shot repetidos — esto es
    lo que se grafica/reporta, NO la `curr_policy` instantánea que puede
    oscilar.
    """

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)

        # RNG LOCAL al agente, no contamina el estado global de NumPy.
        self._rng = np.random.default_rng(seed)

        # Si no se da una política inicial, arranca con la uniforme.
        if initial is None:
            self.curr_policy = np.full(self.game.num_actions(self.agent), 1 / self.game.num_actions(self.agent))
        else:
            self.curr_policy = initial.copy()

        # Regrets acumulados, una entrada por acción del agente.
        self.cum_regrets = np.zeros(self.game.num_actions(self.agent))

        # Suma de políticas instantáneas a lo largo del tiempo, para
        # calcular `learned_policy` como promedio. Arranca con la
        # política inicial ya contada (de ahí `niter = 1`).
        self.sum_policy = self.curr_policy.copy()

        # Política aprendida = promedio temporal. Es lo que converge a
        # un equilibrio correlacionado; `curr_policy` puede oscilar
        # aunque `learned_policy` ya haya convergido.
        self.learned_policy = self.curr_policy.copy()

        # Contador de iteraciones para el promedio. Arranca en 1 porque
        # `sum_policy` ya tiene la política inicial contada.
        self.niter = 1

    def regrets(self, played_actions: ActionDict) -> np.ndarray:
        # Acción que el agente realmente jugó (pivote del regret).
        actions = played_actions.copy()
        a = actions[self.agent]

        # Se clona el juego para simular cada acción alternativa fijando
        # las acciones del RESTO a lo que efectivamente jugaron
        # (semántica contrafáctica).
        g = self.game.clone()
        u = np.zeros(g.num_actions(self.agent), dtype=float)

        # Para cada acción alternativa propia, se simula y se lee la
        # recompensa contrafáctica.
        for alt in g.action_iter(self.agent):
            actions[self.agent] = alt
            g.reset()
            _, rewards, _, _, _ = g.step(actions)
            u[alt] = rewards[self.agent]

        # Regret = utilidad contrafáctica − utilidad efectivamente obtenida.
        return u - u[a]

    def regret_matching(self):
        # Sólo los regrets POSITIVOS importan para la política.
        pos = np.maximum(self.cum_regrets, 0.0)
        total = pos.sum()

        if total > 0:
            # Política proporcional al regret positivo acumulado.
            self.curr_policy = pos / total
        else:
            # Si todos los regrets son no positivos, política uniforme
            # (caso típico al principio, evita 0/0).
            n = self.game.num_actions(self.agent)
            self.curr_policy = np.full(n, 1.0 / n)

        # Acumular en `sum_policy` para promediar luego en `learned_policy`.
        self.sum_policy += self.curr_policy

    def update(self) -> None:
        # `game.observe(self.agent)` devuelve el diccionario de acciones
        # conjuntas de la última ronda. None si todavía no se jugó nada.
        actions = self.game.observe(self.agent)
        if actions is None:
            return

        regrets = self.regrets(actions)
        self.cum_regrets += regrets
        self.regret_matching()
        self.niter += 1
        self.learned_policy = self.sum_policy / self.niter

    def action(self):
        # Incorporar la información de la ronda anterior y muestrear de
        # la política actual. Más legible que `argmax(multinomial(...))`.
        self.update()
        return int(self._rng.choice(len(self.curr_policy), p=self.curr_policy))

    def policy(self):
        # Promedio temporal `learned_policy`, no `curr_policy`. Por
        # ejemplo en RPS `curr_policy` puede oscilar entre concentrarse
        # en cada acción mientras `learned_policy` converge a (1/3, 1/3, 1/3).
        return self.learned_policy
