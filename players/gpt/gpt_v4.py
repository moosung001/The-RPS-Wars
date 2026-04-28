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
        self.my_m1 = {}
        self.my_m2 = {}
        self.my_to_opp = {}
        self.pair_m1 = {}
        self.pair_m2 = {}
        self.outcome_to_opp = {}

        self.source_names = [
            "opp_last",
            "opp_freq",
            "opp_recent",
            "opp_m1",
            "opp_m2",
            "opp_suffix_s",
            "opp_suffix_l",
            "copy_my_last",
            "beat_my_last",
            "lose_my_last",
            "my_next_freq",
            "my_next_recent",
            "my_m1",
            "my_m2",
            "my_suffix",
            "my_to_opp",
            "pair_m1",
            "pair_m2",
            "pair_suffix",
            "outcome",
        ]

        self.experts = [(name, rot) for name in self.source_names for rot in (0, 1, 2)]
        self.decays = (0.65, 0.82, 0.93, 0.985)
        self.scores = [[0.0] * len(self.decays) for _ in self.experts]
        self.last_actions = [0] * len(self.experts)

        self.recent_results = []

    def _counter(self, move):
        if move == ROCK:
            return PAPER
        if move == SCISSORS:
            return ROCK
        return SCISSORS

    def _lose_to(self, move):
        if move == ROCK:
            return SCISSORS
        if move == SCISSORS:
            return PAPER
        return ROCK

    def _payoff(self, a, b):
        if a == b:
            return 0
        if self._counter(b) == a:
            return 1
        return -1

    def _argmax_random(self, arr):
        m = max(arr)
        cand = [i for i, x in enumerate(arr) if x == m]
        return random.choice(cand)

    def _decay_add(self, arr, move, gamma=0.86):
        for i in range(3):
            arr[i] *= gamma
        arr[move] += 1.0

    def _record(self, table, key, move):
        if key not in table:
            table[key] = [0, 0, 0]
        table[key][move] += 1

    def _table_pick(self, table, key, fallback):
        counts = table.get(key)
        if counts is None:
            return fallback
        if counts[0] + counts[1] + counts[2] == 0:
            return fallback
        return self._argmax_random(counts)

    def _suffix_pick(self, seq, max_len=8, lookback=220, max_matches=20):
        n = len(seq)
        if n <= 1:
            return None

        best = min(max_len, n - 1)

        for L in range(best, 0, -1):
            pat = seq[n - L:n]
            counts = [0, 0, 0]
            found = 0

            start = max(0, n - L - 1 - lookback)
            for i in range(n - L - 1, start - 1, -1):
                if seq[i:i + L] == pat:
                    nxt = seq[i + L]
                    counts[nxt] += 1
                    found += 1
                    if found >= max_matches:
                        break

            if found:
                return self._argmax_random(counts)

        return None

    def _pair_suffix_pick(self, max_len=6, lookback=180, max_matches=16):
        n = len(self.pair_hist)
        if n <= 1:
            return None

        best = min(max_len, n - 1)

        for L in range(best, 0, -1):
            pat = self.pair_hist[n - L:n]
            counts = [0, 0, 0]
            found = 0

            start = max(0, n - L - 1 - lookback)
            for i in range(n - L - 1, start - 1, -1):
                if self.pair_hist[i:i + L] == pat:
                    nxt_opp = self.pair_hist[i + L][1]
                    counts[nxt_opp] += 1
                    found += 1
                    if found >= max_matches:
                        break

            if found:
                return self._argmax_random(counts)

        return None

    def _last_outcome(self):
        if not self.my_hist:
            return 0
        return self._payoff(self.my_hist[-1], self.opp_hist[-1])

    def _opp_fallback(self):
        if sum(self.opp_counts) == 0:
            return random.randrange(3)
        return self._argmax_random(self.opp_counts)

    def _my_fallback(self):
        if sum(self.my_counts) == 0:
            return random.randrange(3)
        return self._argmax_random(self.my_counts)

    def _predict_opp_from_my_pred(self, pred_my):
        return self._counter(pred_my)

    def _source_predicted_opp(self, name):
        n = len(self.opp_hist)

        if n == 0:
            return random.randrange(3)

        opp_last = self.opp_hist[-1]
        my_last = self.my_hist[-1]
        fb_opp = self._opp_fallback()
        fb_my = self._my_fallback()

        if name == "opp_last":
            return opp_last

        if name == "opp_freq":
            return self._argmax_random(self.opp_counts)

        if name == "opp_recent":
            return self._argmax_random(self.opp_recent)

        if name == "opp_m1":
            return self._table_pick(self.opp_m1, opp_last, fb_opp)

        if name == "opp_m2":
            if n < 2:
                return fb_opp
            return self._table_pick(self.opp_m2, (self.opp_hist[-2], self.opp_hist[-1]), fb_opp)

        if name == "opp_suffix_s":
            x = self._suffix_pick(self.opp_hist, max_len=5, lookback=120, max_matches=14)
            return fb_opp if x is None else x

        if name == "opp_suffix_l":
            x = self._suffix_pick(self.opp_hist, max_len=10, lookback=280, max_matches=28)
            return fb_opp if x is None else x

        if name == "copy_my_last":
            return my_last

        if name == "beat_my_last":
            return self._counter(my_last)

        if name == "lose_my_last":
            return self._lose_to(my_last)

        if name == "my_next_freq":
            pred_my = self._argmax_random(self.my_counts)
            return self._predict_opp_from_my_pred(pred_my)

        if name == "my_next_recent":
            pred_my = self._argmax_random(self.my_recent)
            return self._predict_opp_from_my_pred(pred_my)

        if name == "my_m1":
            pred_my = self._table_pick(self.my_m1, my_last, fb_my)
            return self._predict_opp_from_my_pred(pred_my)

        if name == "my_m2":
            if n < 2:
                pred_my = fb_my
            else:
                pred_my = self._table_pick(self.my_m2, (self.my_hist[-2], self.my_hist[-1]), fb_my)
            return self._predict_opp_from_my_pred(pred_my)

        if name == "my_suffix":
            pred_my = self._suffix_pick(self.my_hist, max_len=9, lookback=250, max_matches=22)
            if pred_my is None:
                pred_my = fb_my
            return self._predict_opp_from_my_pred(pred_my)

        if name == "my_to_opp":
            return self._table_pick(self.my_to_opp, my_last, fb_opp)

        if name == "pair_m1":
            return self._table_pick(self.pair_m1, (my_last, opp_last), fb_opp)

        if name == "pair_m2":
            if n < 2:
                return fb_opp
            key = ((self.my_hist[-2], self.opp_hist[-2]), (self.my_hist[-1], self.opp_hist[-1]))
            return self._table_pick(self.pair_m2, key, fb_opp)

        if name == "pair_suffix":
            x = self._pair_suffix_pick(max_len=7, lookback=220, max_matches=18)
            return fb_opp if x is None else x

        if name == "outcome":
            return self._table_pick(self.outcome_to_opp, self._last_outcome(), fb_opp)

        return fb_opp

    def _rot_action(self, predicted_opp, rot):
        if rot == 0:
            return self._counter(predicted_opp)
        if rot == 1:
            return predicted_opp
        return self._lose_to(predicted_opp)

    def choose(self, history: list) -> int:
        import math

        n = len(self.opp_hist)

        if n == 0:
            return random.randrange(3)

        source_cache = {}
        candidates = []

        for ei, (name, rot) in enumerate(self.experts):
            if name not in source_cache:
                source_cache[name] = self._source_predicted_opp(name)

            predicted_opp = source_cache[name]
            action = self._rot_action(predicted_opp, rot)
            self.last_actions[ei] = action

            for dj, score in enumerate(self.scores[ei]):
                candidates.append((score, action))

        candidates.sort(key=lambda x: x[0], reverse=True)
        topk = candidates[:12]

        masses = [0.0, 0.0, 0.0]
        top_score = topk[0][0]

        for score, action in topk:
            gap = top_score - score
            if gap > 12.0:
                gap = 12.0
            w = math.exp(-1.35 * gap)
            masses[action] += w

        total = sum(masses)
        if total <= 0.0:
            probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        else:
            probs = [m / total for m in masses]

        recent = self.recent_results[-40:]
        recent_avg = (sum(recent) / len(recent)) if recent else 0.0
        ordered = sorted(probs, reverse=True)
        margin = ordered[0] - ordered[1]

        eps = 0.015

        if n < 12:
            eps += 0.18
        elif n < 40:
            eps += 0.10
        elif n < 120:
            eps += 0.04

        if recent_avg < -0.10:
            eps += 0.10
        elif recent_avg < 0.0:
            eps += 0.05
        elif recent_avg > 0.20 and margin > 0.18:
            eps -= 0.01

        if margin < 0.06:
            eps += 0.08
        elif margin < 0.12:
            eps += 0.03

        if eps < 0.02:
            eps = 0.02
        if eps > 0.24:
            eps = 0.24

        probs = [(1.0 - eps) * p + eps / 3.0 for p in probs]

        best = max(range(3), key=lambda i: probs[i])

        if margin > 0.20 and recent_avg > 0.05:
            return best

        r = random.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c:
                return i

        return best

    def update(self, my_move: int, opp_move: int) -> None:
        for ei, action in enumerate(self.last_actions):
            reward = self._payoff(action, opp_move)
            for dj, d in enumerate(self.decays):
                self.scores[ei][dj] = d * self.scores[ei][dj] + reward

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
            key = ((self.my_hist[-2], self.opp_hist[-2]), (self.my_hist[-1], self.opp_hist[-1]))
            self._record(self.pair_m2, key, opp_move)

        self.my_counts[my_move] += 1
        self.opp_counts[opp_move] += 1

        self._decay_add(self.my_recent, my_move, gamma=0.86)
        self._decay_add(self.opp_recent, opp_move, gamma=0.86)

        self.my_hist.append(my_move)
        self.opp_hist.append(opp_move)
        self.pair_hist.append((my_move, opp_move))

        self.recent_results.append(self._payoff(my_move, opp_move))
        if len(self.recent_results) > 200:
            self.recent_results = self.recent_results[-200:]