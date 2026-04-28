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
        self.pair1 = {}
        self.out_opp = {}
        self.out_my = {}

        self.base_names = [
            "opp_last",
            "opp_freq",
            "opp_recent",
            "opp_m1",
            "opp_m2",
            "pair1",
            "out_opp",
            "cycle_up",
            "cycle_down",
            "my_last",
            "my_freq",
            "my_recent",
            "my_m1",
            "my_m2",
            "out_my",
            "beat_my_last",
            "lose_my_last",
        ]

        self.experts = [(name, rot) for name in self.base_names for rot in (0, 1, 2)]
        self.decays = (0.60, 0.82, 0.94, 0.985)
        self.scores = [[0.0] * len(self.decays) for _ in self.experts]
        self.last_actions = [0] * len(self.experts)
        self.recent_results = []

    def _counter(self, move):
        return (move + 2) % 3

    def _lose_to(self, move):
        return (move + 1) % 3

    def _payoff(self, a, b):
        if a == b:
            return 0
        return 1 if self._counter(b) == a else -1

    def _argmax_random(self, arr):
        m = max(arr)
        cand = [i for i, v in enumerate(arr) if v == m]
        return random.choice(cand)

    def _table_pick(self, table, key, fallback):
        row = table.get(key)
        if row is None:
            return fallback
        return self._argmax_random(row)

    def _record(self, table, key, move):
        if key not in table:
            table[key] = [0, 0, 0]
        table[key][move] += 1

    def _decay_add(self, arr, move, gamma=0.84):
        for i in range(3):
            arr[i] *= gamma
        arr[move] += 1.0

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

        fallback_opp = self._argmax_random(self.opp_counts) if sum(self.opp_counts) > 0 else random.randrange(3)
        fallback_my = self._argmax_random(self.my_counts) if sum(self.my_counts) > 0 else random.randrange(3)

        if name == "opp_last":
            return opp_last

        if name == "opp_freq":
            return self._argmax_random(self.opp_counts)

        if name == "opp_recent":
            return self._argmax_random(self.opp_recent)

        if name == "opp_m1":
            return self._table_pick(self.opp_m1, opp_last, fallback_opp)

        if name == "opp_m2":
            if n < 2:
                return fallback_opp
            return self._table_pick(self.opp_m2, (self.opp_hist[-2], self.opp_hist[-1]), fallback_opp)

        if name == "pair1":
            return self._table_pick(self.pair1, (my_last, opp_last), fallback_opp)

        if name == "out_opp":
            return self._table_pick(self.out_opp, self._last_outcome(), fallback_opp)

        if name == "cycle_up":
            return (opp_last + 1) % 3

        if name == "cycle_down":
            return (opp_last + 2) % 3

        if name == "my_last":
            return my_last

        if name == "my_freq":
            return self._argmax_random(self.my_counts)

        if name == "my_recent":
            return self._argmax_random(self.my_recent)

        if name == "my_m1":
            return self._table_pick(self.my_m1, my_last, fallback_my)

        if name == "my_m2":
            if n < 2:
                return fallback_my
            return self._table_pick(self.my_m2, (self.my_hist[-2], self.my_hist[-1]), fallback_my)

        if name == "out_my":
            return self._table_pick(self.out_my, self._last_outcome(), fallback_my)

        if name == "beat_my_last":
            return self._counter(my_last)

        if name == "lose_my_last":
            return self._lose_to(my_last)

        return random.randrange(3)

    def choose(self, history: list) -> int:
        import math

        n = len(self.opp_hist)
        if n == 0:
            return random.randrange(3)

        source_cache = {}
        candidates = []

        for ei, (name, rot) in enumerate(self.experts):
            if name not in source_cache:
                source_cache[name] = self._base_symbol(name)

            x = source_cache[name]
            action = (x + rot) % 3
            self.last_actions[ei] = action

            best_score = max(self.scores[ei])
            mean_score = sum(self.scores[ei]) / len(self.scores[ei])
            candidates.append((best_score + 0.25 * mean_score, action))

        candidates.sort(key=lambda t: t[0], reverse=True)
        top = candidates[:9]
        top_score = top[0][0]

        masses = [0.0, 0.0, 0.0]
        for score, action in top:
            gap = top_score - score
            if gap < 0.0:
                gap = 0.0
            if gap > 9.0:
                gap = 9.0
            w = math.exp(-1.35 * gap)
            masses[action] += w

        total = sum(masses)
        if total <= 0.0:
            probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        else:
            probs = [m / total for m in masses]

        recent = self.recent_results[-30:]
        recent_avg = (sum(recent) / len(recent)) if recent else 0.0

        ordered = sorted(probs, reverse=True)
        margin = ordered[0] - ordered[1]

        eps = 0.05 + max(0.0, -recent_avg) * 0.10

        if n < 15:
            eps += 0.16
        elif n < 60:
            eps += 0.08
        elif n < 150:
            eps += 0.03

        if margin < 0.06:
            eps += 0.08
        elif margin < 0.12:
            eps += 0.03
        elif margin > 0.20 and recent_avg > 0.10:
            eps -= 0.02

        if eps < 0.04:
            eps = 0.04
        if eps > 0.26:
            eps = 0.26

        probs = [(1.0 - eps) * p + eps / 3.0 for p in probs]

        if margin > 0.18 and recent_avg > 0.02 and random.random() < 0.88:
            return max(range(3), key=lambda i: probs[i])

        r = random.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c:
                return i

        return PAPER

    def update(self, my_move: int, opp_move: int) -> None:
        for ei, action in enumerate(self.last_actions):
            reward = self._payoff(action, opp_move)
            for j, d in enumerate(self.decays):
                self.scores[ei][j] = d * self.scores[ei][j] + reward

        n = len(self.opp_hist)

        if n >= 1:
            self._record(self.opp_m1, self.opp_hist[-1], opp_move)
            self._record(self.my_m1, self.my_hist[-1], my_move)
            self._record(self.pair1, (self.my_hist[-1], self.opp_hist[-1]), opp_move)
            self._record(self.out_opp, self._last_outcome(), opp_move)
            self._record(self.out_my, self._last_outcome(), my_move)

        if n >= 2:
            self._record(self.opp_m2, (self.opp_hist[-2], self.opp_hist[-1]), opp_move)
            self._record(self.my_m2, (self.my_hist[-2], self.my_hist[-1]), my_move)

        self.opp_counts[opp_move] += 1
        self.my_counts[my_move] += 1

        self._decay_add(self.opp_recent, opp_move, gamma=0.84)
        self._decay_add(self.my_recent, my_move, gamma=0.84)

        self.my_hist.append(my_move)
        self.opp_hist.append(opp_move)

        self.recent_results.append(self._payoff(my_move, opp_move))
        if len(self.recent_results) > 200:
            self.recent_results = self.recent_results[-200:]