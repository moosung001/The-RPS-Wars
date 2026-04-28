# 내용을 여기에 채워주세요
class GPTPlayer(BasePlayer):
    def __init__(self):
        super().__init__(name="GPT")
        self.reset()

    def reset(self):
        super().reset()

        self.my_hist = []
        self.opp_hist = []

        self.my_counts = [0, 0, 0]
        self.opp_counts = [0, 0, 0]

        self.my_recent = [0.0, 0.0, 0.0]
        self.opp_recent = [0.0, 0.0, 0.0]

        self.opp_m1 = {}
        self.opp_m2 = {}
        self.my_m1 = {}
        self.my_m2 = {}
        self.pair = {}
        self.out_opp = {}
        self.out_my = {}

        self.base_names = [
            "opp_last",
            "opp_freq",
            "opp_recent",
            "opp_m1",
            "opp_m2",
            "pair",
            "out_opp",
            "cycle_up",
            "cycle_down",
            "my_last",
            "my_freq",
            "my_recent",
            "my_m1",
            "my_m2",
            "out_my",
        ]

        self.experts = [(name, rot) for name in self.base_names for rot in (0, 1, 2)]
        self.logw = [0.0] * len(self.experts)
        self.last_actions = [0] * len(self.experts)

        self.recent_results = []
        self.last_probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    def _payoff(self, a, b):
        if (a == ROCK and b == SCISSORS) or (a == SCISSORS and b == PAPER) or (a == PAPER and b == ROCK):
            return 1
        if (b == ROCK and a == SCISSORS) or (b == SCISSORS and a == PAPER) or (b == PAPER and a == ROCK):
            return -1
        return 0

    def _decay_add(self, arr, move, gamma=0.85):
        for i in range(3):
            arr[i] *= gamma
        arr[move] += 1.0

    def _argmax_rand(self, xs):
        m = max(xs)
        cand = [i for i, x in enumerate(xs) if x == m]
        return random.choice(cand)

    def _table_pick(self, table, key, fallback):
        if key not in table:
            return fallback
        return self._argmax_rand(table[key])

    def _last_outcome(self):
        if not self.my_hist:
            return 0
        return self._payoff(self.my_hist[-1], self.opp_hist[-1])

    def _base_symbol(self, name):
        n = len(self.opp_hist)

        if n == 0:
            return random.randrange(3)

        opp_last = self.opp_hist[-1]
        my_last = self.my_hist[-1]

        fallback_opp = self._argmax_rand(self.opp_counts)
        fallback_my = self._argmax_rand(self.my_counts) if sum(self.my_counts) > 0 else random.randrange(3)

        if name == "opp_last":
            return opp_last

        if name == "opp_freq":
            return self._argmax_rand(self.opp_counts)

        if name == "opp_recent":
            return self._argmax_rand(self.opp_recent)

        if name == "opp_m1":
            return self._table_pick(self.opp_m1, opp_last, fallback_opp)

        if name == "opp_m2":
            if n < 2:
                return fallback_opp
            return self._table_pick(self.opp_m2, (self.opp_hist[-2], self.opp_hist[-1]), fallback_opp)

        if name == "pair":
            return self._table_pick(self.pair, (my_last, opp_last), fallback_opp)

        if name == "out_opp":
            return self._table_pick(self.out_opp, self._last_outcome(), fallback_opp)

        if name == "cycle_up":
            return (opp_last + 1) % 3

        if name == "cycle_down":
            return (opp_last + 2) % 3

        if name == "my_last":
            return my_last

        if name == "my_freq":
            return self._argmax_rand(self.my_counts)

        if name == "my_recent":
            return self._argmax_rand(self.my_recent)

        if name == "my_m1":
            return self._table_pick(self.my_m1, my_last, fallback_my)

        if name == "my_m2":
            if n < 2:
                return fallback_my
            return self._table_pick(self.my_m2, (self.my_hist[-2], self.my_hist[-1]), fallback_my)

        if name == "out_my":
            return self._table_pick(self.out_my, self._last_outcome(), fallback_my)

        return random.randrange(3)

    def choose(self, history: list) -> int:
        import math

        n = len(self.opp_hist)
        masses = [0.0, 0.0, 0.0]
        acts = []

        for idx, (name, rot) in enumerate(self.experts):
            x = self._base_symbol(name)
            action = (x + rot) % 3
            acts.append(action)

            lw = self.logw[idx]
            if lw > 8.0:
                lw = 8.0
            elif lw < -8.0:
                lw = -8.0

            w = math.exp(lw)
            masses[action] += w

        self.last_actions = acts

        s = sum(masses)
        if s <= 0:
            probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        else:
            probs = [m / s for m in masses]

        recent_window = self.recent_results[-30:]
        recent_avg = sum(recent_window) / len(recent_window) if recent_window else 0.0
        spread = max(probs) - min(probs)

        eps = 0.06 + max(0.0, -recent_avg) * 0.45

        if n < 15:
            eps += 0.18
        elif n < 60:
            eps += 0.10
        elif n < 150:
            eps += 0.05

        if spread < 0.12:
            eps += 0.08

        if eps < 0.08:
            eps = 0.08
        elif eps > 0.35:
            eps = 0.35

        probs = [(1.0 - eps) * p + eps / 3.0 for p in probs]
        self.last_probs = probs

        r = random.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c:
                return i

        return PAPER

    def update(self, my_move: int, opp_move: int) -> None:
        import math

        eta = 0.22
        decay = 0.97

        for i, action in enumerate(self.last_actions):
            reward = self._payoff(action, opp_move)
            self.logw[i] = decay * self.logw[i] + eta * reward

        n = len(self.opp_hist)

        if n >= 1:
            self.opp_m1.setdefault(self.opp_hist[-1], [0, 0, 0])[opp_move] += 1
            self.my_m1.setdefault(self.my_hist[-1], [0, 0, 0])[my_move] += 1
            self.pair.setdefault((self.my_hist[-1], self.opp_hist[-1]), [0, 0, 0])[opp_move] += 1

            prev_outcome = self._last_outcome()
            self.out_opp.setdefault(prev_outcome, [0, 0, 0])[opp_move] += 1
            self.out_my.setdefault(prev_outcome, [0, 0, 0])[my_move] += 1

        if n >= 2:
            self.opp_m2.setdefault((self.opp_hist[-2], self.opp_hist[-1]), [0, 0, 0])[opp_move] += 1
            self.my_m2.setdefault((self.my_hist[-2], self.my_hist[-1]), [0, 0, 0])[my_move] += 1

        self.opp_counts[opp_move] += 1
        self.my_counts[my_move] += 1

        self._decay_add(self.opp_recent, opp_move, gamma=0.85)
        self._decay_add(self.my_recent, my_move, gamma=0.85)

        self.my_hist.append(my_move)
        self.opp_hist.append(opp_move)

        self.recent_results.append(self._payoff(my_move, opp_move))
        if len(self.recent_results) > 200:
            self.recent_results = self.recent_results[-200:]