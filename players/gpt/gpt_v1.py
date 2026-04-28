class GPTPlayer(BasePlayer):
    def __init__(self):
        super().__init__(name="GPT")
        self.reset()

    def reset(self):
        super().reset()
        self.my_hist = []
        self.opp_hist = []

        self.opp_counts = [0, 0, 0]
        self.markov1 = {}
        self.markov2 = {}
        self.by_pair = {}
        self.by_my_last = {}

        self.model_scores = {}
        self.last_model_probs = {}

    def _one_hot_probs(self, move, sharp=0.86):
        off = (1.0 - sharp) / 2.0
        p = [off, off, off]
        p[move] = sharp
        return p

    def _normalize(self, xs):
        s = float(sum(xs))
        if s <= 0:
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        return [x / s for x in xs]

    def _smoothed_probs(self, counts, alpha=1.0):
        total = counts[0] + counts[1] + counts[2] + 3.0 * alpha
        return [(counts[i] + alpha) / total for i in range(3)]

    def _blend(self, p, q, w=0.75):
        return [w * p[i] + (1.0 - w) * q[i] for i in range(3)]

    def _get_table_probs(self, table, key, backoff, alpha=1.0):
        counts = table.get(key)
        if counts is None:
            return list(backoff)
        return self._blend(self._smoothed_probs(counts, alpha), backoff, 0.75)

    def _recent_probs(self, horizon=20, gamma=0.85):
        if not self.opp_hist:
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

        counts = [0.0, 0.0, 0.0]
        weight = 1.0
        for move in reversed(self.opp_hist[-horizon:]):
            counts[move] += weight
            weight *= gamma

        total = sum(counts) + 3.0
        return [(counts[i] + 1.0) / total for i in range(3)]

    def _record(self, table, key, nxt_move):
        if key not in table:
            table[key] = [0, 0, 0]
        table[key][nxt_move] += 1

    def choose(self, history) -> int:
        import math

        n = len(self.opp_hist)

        global_p = self._smoothed_probs(self.opp_counts, alpha=1.0)
        recent_p = self._recent_probs()
        base_p = self._blend(recent_p, global_p, 0.65)

        models = {
            "global": global_p,
            "recent": recent_p,
        }

        if n >= 1:
            opp_last = self.opp_hist[-1]
            my_last = self.my_hist[-1]

            models["markov1_opp"] = self._get_table_probs(self.markov1, opp_last, base_p)
            models["by_my_last"] = self._get_table_probs(self.by_my_last, my_last, base_p)
            models["pair_last"] = self._get_table_probs(self.by_pair, (my_last, opp_last), base_p)

            models["repeat_opp"] = self._one_hot_probs(opp_last, 0.86)
            models["cycle_up"] = self._one_hot_probs((opp_last + 1) % 3, 0.86)
            models["cycle_down"] = self._one_hot_probs((opp_last + 2) % 3, 0.86)
            models["copy_me"] = self._one_hot_probs(my_last, 0.86)
            models["beat_my_last"] = self._one_hot_probs((my_last + 2) % 3, 0.86)
            models["lose_to_my_last"] = self._one_hot_probs((my_last + 1) % 3, 0.86)

        if n >= 2:
            models["markov2_opp"] = self._get_table_probs(
                self.markov2,
                (self.opp_hist[-2], self.opp_hist[-1]),
                base_p
            )

        mixed = [0.0, 0.0, 0.0]
        for name, probs in models.items():
            score = self.model_scores.get(name, 0.0)
            weight = math.exp(max(-4.0, min(4.0, score)) * 1.6)
            for i in range(3):
                mixed[i] += weight * probs[i]

        p_opp = self._normalize(mixed)

        if n < 3:
            eps = 0.22
        elif n < 8:
            eps = 0.12
        elif n < 20:
            eps = 0.05
        else:
            eps = 0.02

        p_opp = [(1.0 - eps) * p_opp[i] + eps / 3.0 for i in range(3)]
        self.last_model_probs = models

        payoffs = [
            p_opp[1] - p_opp[2],
            p_opp[2] - p_opp[0],
            p_opp[0] - p_opp[1],
        ]

        if max(p_opp) - min(p_opp) < 0.04:
            return random.choice([0, 1, 2])

        best_value = max(payoffs)
        candidates = [i for i, v in enumerate(payoffs) if abs(v - best_value) < 1e-12]
        return random.choice(candidates)

    def update(self, my_move: int, opp_move: int) -> None:
        for name, probs in self.last_model_probs.items():
            reward = probs[opp_move] - (1.0 / 3.0)
            old = self.model_scores.get(name, 0.0)
            self.model_scores[name] = 0.90 * old + 1.35 * reward

        n = len(self.opp_hist)

        if n >= 1:
            self._record(self.markov1, self.opp_hist[-1], opp_move)
            self._record(self.by_my_last, self.my_hist[-1], opp_move)
            self._record(self.by_pair, (self.my_hist[-1], self.opp_hist[-1]), opp_move)

        if n >= 2:
            self._record(self.markov2, (self.opp_hist[-2], self.opp_hist[-1]), opp_move)

        self.opp_counts[opp_move] += 1
        self.my_hist.append(my_move)
        self.opp_hist.append(opp_move)