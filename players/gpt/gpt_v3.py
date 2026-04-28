# 내용을 여기에 채워주세요
class GPTPlayer(BasePlayer):
    def __init__(self):
        super().__init__(name="GPT")
        self.reset()

    def reset(self):
        super().reset()

        self.my_hist = []
        self.opp_hist = []
        self.pair_hist = []

        self.my_counts = [0, 0, 0]
        self.opp_counts = [0, 0, 0]

        self.my_recent = [0.0, 0.0, 0.0]
        self.opp_recent = [0.0, 0.0, 0.0]

        self.opp_m1 = {}
        self.opp_m2 = {}
        self.opp_m3 = {}

        self.my_m1 = {}
        self.my_m2 = {}

        self.my_to_opp = {}
        self.pair_m1 = {}
        self.pair_m2 = {}
        self.outcome_to_opp = {}

        self.source_names = [
            "opp_last",
            "opp_mode",
            "opp_recent",
            "opp_m1",
            "opp_m2",
            "opp_m3",
            "opp_suffix",
            "my_last",
            "my_mode",
            "my_recent",
            "my_m1",
            "my_m2",
            "my_suffix",
            "my_to_opp",
            "pair_m1",
            "pair_m2",
            "pair_suffix",
            "outcome",
            "cycle_up",
            "cycle_down",
        ]

        self.experts = [(name, rot) for name in self.source_names for rot in (0, 1, 2)]
        self.decays = (0.55, 0.80, 0.93, 0.985)
        self.scores = [[0.0] * len(self.decays) for _ in self.experts]
        self.last_actions = [0] * len(self.experts)

        self.recent_results = []

    def _payoff(self, a, b):
        if (a == ROCK and b == SCISSORS) or (a == SCISSORS and b == PAPER) or (a == PAPER and b == ROCK):
            return 1
        if (b == ROCK and a == SCISSORS) or (b == SCISSORS and a == PAPER) or (b == PAPER and a == ROCK):
            return -1
        return 0

    def _argmax_random(self, arr):
        m = max(arr)
        cand = [i for i, x in enumerate(arr) if x == m]
        return random.choice(cand)

    def _decay_add(self, arr, move, gamma=0.85):
        for i in range(3):
            arr[i] *= gamma
        arr[move] += 1.0

    def _record(self, table, key, val):
        if key not in table:
            table[key] = [0, 0, 0]
        table[key][val] += 1

    def _table_pick(self, table, key, fallback):
        counts = table.get(key)
        if counts is None or (counts[0] + counts[1] + counts[2] == 0):
            return fallback
        return self._argmax_random(counts)

    def _suffix_pick(self, seq, max_len=10, lookback=250, max_matches=24):
        n = len(seq)
        if n <= 1:
            return None

        max_len = min(max_len, n - 1)

        for L in range(max_len, 0, -1):
            pattern = seq[n - L:n]
            counts = [0, 0, 0]
            found = 0

            start = max(0, n - L - 1 - lookback)
            for i in range(n - L - 1, start - 1, -1):
                if seq[i:i + L] == pattern:
                    nxt = seq[i + L]
                    counts[nxt] += 1
                    found += 1
                    if found >= max_matches:
                        break

            if found > 0:
                return self._argmax_random(counts)

        return None

    def _pair_suffix_pick(self, max_len=8, lookback=220, max_matches=20):
        n = len(self.pair_hist)
        if n <= 1:
            return None

        max_len = min(max_len, n - 1)

        for L in range(max_len, 0, -1):
            pattern = self.pair_hist[n - L:n]
            counts = [0, 0, 0]
            found = 0

            start = max(0, n - L - 1 - lookback)
            for i in range(n - L - 1, start - 1, -1):
                if self.pair_hist[i:i + L] == pattern:
                    nxt_opp = self.pair_hist[i + L][1]
                    counts[nxt_opp] += 1
                    found += 1
                    if found >= max_matches:
                        break

            if found > 0:
                return self._argmax_random(counts)

        return None

    def _last_outcome(self):
        if not self.my_hist:
            return 0
        return self._payoff(self.my_hist[-1], self.opp_hist[-1])

    def _fallback_opp(self):
        if self.opp_counts[0] + self.opp_counts[1] + self.opp_counts[2] == 0:
            return random.randrange(3)
        return self._argmax_random(self.opp_counts)

    def _fallback_my(self):
        if self.my_counts[0] + self.my_counts[1] + self.my_counts[2] == 0:
            return random.randrange(3)
        return self._argmax_random(self.my_counts)

    def _source_symbol(self, name):
        n = len(self.opp_hist)

        if n == 0:
            return random.randrange(3)

        opp_last = self.opp_hist[-1]
        my_last = self.my_hist[-1]

        fb_opp = self._fallback_opp()
        fb_my = self._fallback_my()

        if name == "opp_last":
            return opp_last

        if name == "opp_mode":
            return self._argmax_random(self.opp_counts)

        if name == "opp_recent":
            return self._argmax_random(self.opp_recent)

        if name == "opp_m1":
            return self._table_pick(self.opp_m1, opp_last, fb_opp)

        if name == "opp_m2":
            if n < 2:
                return fb_opp
            return self._table_pick(self.opp_m2, (self.opp_hist[-2], self.opp_hist[-1]), fb_opp)

        if name == "opp_m3":
            if n < 3:
                return fb_opp
            return self._table_pick(
                self.opp_m3,
                (self.opp_hist[-3], self.opp_hist[-2], self.opp_hist[-1]),
                fb_opp,
            )

        if name == "opp_suffix":
            x = self._suffix_pick(self.opp_hist, max_len=10, lookback=250, max_matches=24)
            return fb_opp if x is None else x

        if name == "my_last":
            return my_last

        if name == "my_mode":
            return self._argmax_random(self.my_counts)

        if name == "my_recent":
            return self._argmax_random(self.my_recent)

        if name == "my_m1":
            return self._table_pick(self.my_m1, my_last, fb_my)

        if name == "my_m2":
            if n < 2:
                return fb_my
            return self._table_pick(self.my_m2, (self.my_hist[-2], self.my_hist[-1]), fb_my)

        if name == "my_suffix":
            x = self._suffix_pick(self.my_hist, max_len=10, lookback=250, max_matches=24)
            return fb_my if x is None else x

        if name == "my_to_opp":
            return self._table_pick(self.my_to_opp, my_last, fb_opp)

        if name == "pair_m1":
            return self._table_pick(self.pair_m1, (my_last, opp_last), fb_opp)

        if name == "pair_m2":
            if n < 2:
                return fb_opp
            return self._table_pick(
                self.pair_m2,
                (self.pair_hist[-2], self.pair_hist[-1]),
                fb_opp,
            )

        if name == "pair_suffix":
            x = self._pair_suffix_pick(max_len=8, lookback=220, max_matches=20)
            return fb_opp if x is None else x

        if name == "outcome":
            return self._table_pick(self.outcome_to_opp, self._last_outcome(), fb_opp)

        if name == "cycle_up":
            return (opp_last + 1) % 3

        if name == "cycle_down":
            return (opp_last + 2) % 3

        return random.randrange(3)

    def choose(self, history: list) -> int:
        import math

        n = len(self.opp_hist)
        source_cache = {}
        action_mass = [0.0, 0.0, 0.0]

        for idx, (name, rot) in enumerate(self.experts):
            if name not in source_cache:
                source_cache[name] = self._source_symbol(name)

            symbol = source_cache[name]
            action = (symbol + rot) % 3
            self.last_actions[idx] = action

            effective_score = max(self.scores[idx])
            if effective_score > 12.0:
                effective_score = 12.0
            elif effective_score < -12.0:
                effective_score = -12.0

            weight = math.exp(0.80 * effective_score)
            action_mass[action] += weight

        total = action_mass[0] + action_mass[1] + action_mass[2]
        if total <= 0.0:
            probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        else:
            probs = [action_mass[i] / total for i in range(3)]

        recent = self.recent_results[-50:]
        recent_avg = (sum(recent) / len(recent)) if recent else 0.0

        ordered = sorted(probs, reverse=True)
        margin = ordered[0] - ordered[1]

        eps = 0.02

        if n < 15:
            eps += 0.18
        elif n < 60:
            eps += 0.10
        elif n < 150:
            eps += 0.05

        if recent_avg < 0.0:
            eps += min(0.18, -recent_avg * 0.18)

        if margin < 0.05:
            eps += 0.12
        elif margin < 0.10:
            eps += 0.06
        elif margin > 0.22 and recent_avg > 0.10:
            eps -= 0.02

        if eps < 0.03:
            eps = 0.03
        elif eps > 0.28:
            eps = 0.28

        probs = [(1.0 - eps) * p + eps / 3.0 for p in probs]

        r = random.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c:
                return i

        return PAPER

    def update(self, my_move: int, opp_move: int) -> None:
        for i, action in enumerate(self.last_actions):
            reward = self._payoff(action, opp_move)
            for j, d in enumerate(self.decays):
                self.scores[i][j] = d * self.scores[i][j] + reward

        n = len(self.opp_hist)

        if n >= 1:
            self._record(self.opp_m1, self.opp_hist[-1], opp_move)
            self._record(self.my_m1, self.my_hist[-1], my_move)
            self._record(self.my_to_opp, self.my_hist[-1], opp_move)
            self._record(self.pair_m1, (self.my_hist[-1], self.opp_hist[-1]), opp_move)
            self._record(self.outcome_to_opp, self._last_outcome(), opp_move)

        if n >= 2:
            self._record(self.opp_m2, (self.opp_hist[-2], self.opp_hist[-1]), opp_move)
            self._record(self.my_m2, (self.my_hist[-2], self.my_hist[-1]), my_move)
            self._record(self.pair_m2, (self.pair_hist[-2], self.pair_hist[-1]), opp_move)

        if n >= 3:
            self._record(
                self.opp_m3,
                (self.opp_hist[-3], self.opp_hist[-2], self.opp_hist[-1]),
                opp_move,
            )

        self.my_counts[my_move] += 1
        self.opp_counts[opp_move] += 1

        self._decay_add(self.my_recent, my_move, gamma=0.85)
        self._decay_add(self.opp_recent, opp_move, gamma=0.85)

        self.my_hist.append(my_move)
        self.opp_hist.append(opp_move)
        self.pair_hist.append((my_move, opp_move))

        self.recent_results.append(self._payoff(my_move, opp_move))
        if len(self.recent_results) > 200:
            self.recent_results = self.recent_results[-200:]