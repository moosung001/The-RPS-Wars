class GPTPlayer(BasePlayer):
    class _ModeGPTV2:
        def __init__(self, seed):
            self.rng = random.Random(seed)
            self.reset()

        def reset(self):
            self.history = []
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
                "opp_last", "opp_freq", "opp_recent", "opp_m1", "opp_m2",
                "pair", "out_opp", "cycle_up", "cycle_down",
                "my_last", "my_freq", "my_recent", "my_m1", "my_m2", "out_my",
            ]
            self.experts = [(name, rot) for name in self.base_names for rot in (0, 1, 2)]
            self.logw = [0.0] * len(self.experts)
            self.last_actions = [0] * len(self.experts)
            self.recent_results = []

        def _payoff(self, a, b):
            if a == b:
                return 0
            return 1 if (a - b) % 3 == 2 else -1

        def _decay_add(self, arr, move, gamma=0.85):
            for i in range(3):
                arr[i] *= gamma
            arr[move] += 1.0

        def _argmax_rand(self, xs):
            m = max(xs)
            cand = [i for i, x in enumerate(xs) if x == m]
            return self.rng.choice(cand)

        def _table_pick(self, table, key, fallback):
            row = table.get(key)
            return fallback if row is None else self._argmax_rand(row)

        def _last_outcome(self):
            if not self.my_hist:
                return 0
            return self._payoff(self.my_hist[-1], self.opp_hist[-1])

        def _base_symbol(self, name):
            n = len(self.opp_hist)
            if n == 0:
                return self.rng.randrange(3)

            opp_last = self.opp_hist[-1]
            my_last = self.my_hist[-1]

            fallback_opp = self._argmax_rand(self.opp_counts)
            fallback_my = self._argmax_rand(self.my_counts) if sum(self.my_counts) > 0 else self.rng.randrange(3)

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
            return self.rng.randrange(3)

        def choose(self, history):
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
                masses[action] += math.exp(lw)

            self.last_actions = acts

            total = sum(masses)
            probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0] if total <= 0 else [m / total for m in masses]

            recent = self.recent_results[-30:]
            recent_avg = sum(recent) / len(recent) if recent else 0.0
            spread = max(probs) - min(probs)

            eps = 0.06 + max(0.0, -recent_avg) * 0.45
            n = len(self.opp_hist)
            if n < 15:
                eps += 0.18
            elif n < 60:
                eps += 0.10
            elif n < 150:
                eps += 0.05
            if spread < 0.12:
                eps += 0.08
            eps = max(0.08, min(0.35, eps))

            probs = [(1.0 - eps) * p + eps / 3.0 for p in probs]

            r = self.rng.random()
            c = 0.0
            for i, p in enumerate(probs):
                c += p
                if r <= c:
                    return i
            return 2

        def update(self, my_move, opp_move):
            for i, action in enumerate(self.last_actions):
                self.logw[i] = 0.97 * self.logw[i] + 0.22 * self._payoff(action, opp_move)

            n = len(self.opp_hist)
            if n >= 1:
                self.opp_m1.setdefault(self.opp_hist[-1], [0, 0, 0])[opp_move] += 1
                self.my_m1.setdefault(self.my_hist[-1], [0, 0, 0])[my_move] += 1
                self.pair.setdefault((self.my_hist[-1], self.opp_hist[-1]), [0, 0, 0])[opp_move] += 1

                prev = self._last_outcome()
                self.out_opp.setdefault(prev, [0, 0, 0])[opp_move] += 1
                self.out_my.setdefault(prev, [0, 0, 0])[my_move] += 1

            if n >= 2:
                self.opp_m2.setdefault((self.opp_hist[-2], self.opp_hist[-1]), [0, 0, 0])[opp_move] += 1
                self.my_m2.setdefault((self.my_hist[-2], self.my_hist[-1]), [0, 0, 0])[my_move] += 1

            self.opp_counts[opp_move] += 1
            self.my_counts[my_move] += 1
            self._decay_add(self.opp_recent, opp_move)
            self._decay_add(self.my_recent, my_move)

            self.my_hist.append(my_move)
            self.opp_hist.append(opp_move)
            self.history.append((my_move, opp_move))

            self.recent_results.append(self._payoff(my_move, opp_move))
            if len(self.recent_results) > 200:
                self.recent_results = self.recent_results[-200:]

    class _ShadowGeminiV3:
        def __init__(self):
            self.reset()

        def reset(self):
            self.history = []
            self.my_history = []
            self.op_history = []
            self.scores = [0.0] * 49
            self.last_predictions = [0] * 49

        def predict_next(self, seq, d):
            if len(seq) <= d:
                return None
            target = seq[-d:]
            for i in range(len(seq) - d - 1, -1, -1):
                if seq[i:i + d] == target:
                    return seq[i + d]
            return None

        def predict_next_comb(self, comb, d):
            if len(comb) <= d:
                return None
            target = comb[-d:]
            for i in range(len(comb) - d - 1, -1, -1):
                if comb[i:i + d] == target:
                    return self.op_history[i + d]
            return None

        def choose(self, history):
            if not self.my_history:
                move = random.randint(0, 2)
                self.last_predictions = [move] * 49
                return move

            preds = []

            for d in range(1, 6):
                p = self.predict_next(self.op_history, d)
                if p is None:
                    p = random.randint(0, 2)
                preds.extend([(p + 2) % 3, p, (p + 1) % 3])

            for d in range(1, 6):
                p = self.predict_next(self.my_history, d)
                if p is None:
                    p = random.randint(0, 2)
                preds.extend([(p + 2) % 3, p, (p + 1) % 3])

            comb = list(zip(self.my_history, self.op_history))
            for d in range(1, 6):
                p = self.predict_next_comb(comb, d)
                if p is None:
                    p = random.randint(0, 2)
                preds.extend([(p + 2) % 3, p, (p + 1) % 3])

            counts = [self.op_history.count(0), self.op_history.count(1), self.op_history.count(2)]
            p = counts.index(max(counts))
            preds.extend([(p + 2) % 3, p, (p + 1) % 3])
            preds.append(random.randint(0, 2))

            self.last_predictions = preds
            best = max(self.scores)
            if best <= 0:
                return random.randint(0, 2)
            return preds[self.scores.index(best)]

        def update(self, my_move, opp_move):
            for i, p in enumerate(self.last_predictions):
                self.scores[i] *= 0.90
                if p == (opp_move + 2) % 3:
                    self.scores[i] += 1.2
                elif p == opp_move:
                    self.scores[i] -= 0.5
                else:
                    self.scores[i] -= 1.0

            self.my_history.append(my_move)
            self.op_history.append(opp_move)
            self.history.append((my_move, opp_move))

    class _ShadowGeminiV4:
        def __init__(self):
            self.reset()

        def reset(self):
            self.history = []
            self.my_history = []
            self.op_history = []
            self.weights = [1.0] * 13
            self.last_predictions = [0] * 13

        def get_counter(self, move):
            return (move - 1) % 3

        def find_longest_match(self, seq, max_depth=20, lookback=260):
            n = len(seq)
            if n < 2:
                return None
            for d in range(min(max_depth, n - 1), 0, -1):
                target = seq[n - d:n]
                start = max(0, n - d - 1 - lookback)
                for i in range(n - d - 1, start - 1, -1):
                    if seq[i:i + d] == target:
                        return seq[i + d]
            return None

        def find_longest_comb_match(self, comb, max_depth=20, lookback=220):
            n = len(comb)
            if n < 2:
                return None
            for d in range(min(max_depth, n - 1), 0, -1):
                target = comb[n - d:n]
                start = max(0, n - d - 1 - lookback)
                for i in range(n - d - 1, start - 1, -1):
                    if comb[i:i + d] == target:
                        return self.op_history[i + d]
            return None

        def choose(self, history):
            if not self.my_history:
                move = random.randint(0, 2)
                self.last_predictions = [move] * 13
                return move

            preds = []

            p_op = self.find_longest_match(self.op_history)
            if p_op is None:
                p_op = random.randint(0, 2)
            preds.extend([self.get_counter(p_op), p_op, (p_op + 1) % 3])

            p_my = self.find_longest_match(self.my_history)
            if p_my is None:
                p_my = random.randint(0, 2)
            opp_guess = self.get_counter(p_my)
            preds.extend([self.get_counter(opp_guess), opp_guess, (opp_guess + 1) % 3])

            comb = list(zip(self.my_history, self.op_history))
            p_comb = self.find_longest_comb_match(comb)
            if p_comb is None:
                p_comb = random.randint(0, 2)
            preds.extend([self.get_counter(p_comb), p_comb, (p_comb + 1) % 3])

            counts = [self.op_history.count(0), self.op_history.count(1), self.op_history.count(2)]
            if sum(counts) > 0:
                p_freq = counts.index(max(counts))
            else:
                p_freq = random.randint(0, 2)
            preds.extend([self.get_counter(p_freq), p_freq, (p_freq + 1) % 3])

            preds.append(random.randint(0, 2))
            self.last_predictions = preds

            total = sum(self.weights)
            r = random.uniform(0.0, total)
            cum = 0.0
            for i, w in enumerate(self.weights):
                cum += w
                if r <= cum:
                    return preds[i]
            return preds[-1]

        def update(self, my_move, opp_move):
            for i, p in enumerate(self.last_predictions):
                self.weights[i] *= 0.96
                if p == self.get_counter(opp_move):
                    self.weights[i] *= 1.15
                elif p == opp_move:
                    self.weights[i] *= 0.90
                else:
                    self.weights[i] *= 0.50
                if self.weights[i] < 0.01:
                    self.weights[i] = 0.01

            self.my_history.append(my_move)
            self.op_history.append(opp_move)
            self.history.append((my_move, opp_move))

    class _ShadowGeminiV5:
        def __init__(self):
            self.reset()

        def reset(self):
            import numpy as np
            self.history = []
            self.my_history = []
            self.op_history = []
            self.max_depth = 6
            self.num_engines = 58
            self.weights = np.zeros(self.num_engines)
            self.lr = 0.5
            self.decay = 0.98
            self.last_probs = np.ones((self.num_engines, 3)) / 3.0

        def get_dist(self, seq, depth):
            import numpy as np
            dist = np.ones(3) * 0.1
            if len(seq) <= depth:
                return dist
            target = tuple(seq[-depth:])
            start = max(0, len(seq) - depth - 1 - 320)
            for i in range(len(seq) - depth - 1, start - 1, -1):
                if tuple(seq[i:i + depth]) == target:
                    dist[self.op_history[i + depth]] += 1.0
            return dist

        def get_comb_dist(self, comb, depth):
            import numpy as np
            dist = np.ones(3) * 0.1
            if len(comb) <= depth:
                return dist
            target = tuple(comb[-depth:])
            start = max(0, len(comb) - depth - 1 - 280)
            for i in range(len(comb) - depth - 1, start - 1, -1):
                if tuple(comb[i:i + depth]) == target:
                    dist[self.op_history[i + depth]] += 1.0
            return dist

        def choose(self, history):
            import numpy as np
            if not self.my_history:
                return random.randint(0, 2)

            all_probs = []

            for d in range(1, self.max_depth + 1):
                dist = self.get_dist(self.op_history, d)
                p = dist / np.sum(dist)
                all_probs.extend([np.roll(p, 0), np.roll(p, 1), np.roll(p, 2)])

            for d in range(1, self.max_depth + 1):
                dist = self.get_dist(self.my_history, d)
                p = dist / np.sum(dist)
                all_probs.extend([np.roll(p, 0), np.roll(p, 1), np.roll(p, 2)])

            comb = list(zip(self.my_history, self.op_history))
            for d in range(1, self.max_depth + 1):
                dist = self.get_comb_dist(comb, d)
                p = dist / np.sum(dist)
                all_probs.extend([np.roll(p, 0), np.roll(p, 1), np.roll(p, 2)])

            counts = np.array([self.op_history.count(0), self.op_history.count(1), self.op_history.count(2)]) + 0.1
            p_freq = counts / np.sum(counts)
            all_probs.extend([np.roll(p_freq, 0), np.roll(p_freq, 1), np.roll(p_freq, 2)])
            all_probs.append(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))

            self.last_probs = np.array(all_probs)

            exp_w = np.exp(self.weights - np.max(self.weights))
            meta_probs = exp_w / np.sum(exp_w)
            final_opp_probs = np.sum(self.last_probs * meta_probs[:, None], axis=0)

            predicted_opp = int(np.argmax(final_opp_probs))
            return int((predicted_opp - 1) % 3)

        def update(self, my_move, opp_move):
            import numpy as np
            if self.my_history:
                actual_probs = self.last_probs[:, opp_move]
                actual_probs = np.clip(actual_probs, 1e-7, 1.0)
                self.weights = self.weights * self.decay + self.lr * np.log(actual_probs)

            self.my_history.append(my_move)
            self.op_history.append(opp_move)
            self.history.append((my_move, opp_move))

    class _ModeSafe:
        def __init__(self, seed):
            self.rng = random.Random(seed)
            self.reset()

        def reset(self):
            self.history = []
            self.my_counts = [0, 0, 0]
            self.opp_counts = [0, 0, 0]
            self.my_recent = [0.0, 0.0, 0.0]
            self.opp_recent = [0.0, 0.0, 0.0]

        def _counter(self, move):
            return (move + 2) % 3

        def _argmax_rand(self, arr):
            m = max(arr)
            cand = [i for i, x in enumerate(arr) if x == m]
            return self.rng.choice(cand)

        def choose(self, history):
            if not history:
                return self.rng.randrange(3)

            my_last = history[-1][0]
            opp_last = history[-1][1]
            masses = [1.0, 1.0, 1.0]

            if sum(self.opp_counts) > 0:
                pred_freq = self._argmax_rand(self.opp_counts)
                masses[self._counter(pred_freq)] += 0.35

            pred_recent = self._argmax_rand(self.opp_recent)
            masses[self._counter(pred_recent)] += 0.55
            masses[self._counter(opp_last)] += 0.25

            total_my = sum(self.my_counts)
            target = (total_my + 1) / 3.0

            for a in range(3):
                masses[a] += 0.15 * (target - self.my_counts[a])
                if a == my_last:
                    masses[a] -= 0.05
                if len(history) >= 2 and a == history[-1][0] == history[-2][0]:
                    masses[a] -= 0.15
                if masses[a] < 0.05:
                    masses[a] = 0.05

            total = sum(masses)
            probs = [m / total for m in masses]

            r = self.rng.random()
            c = 0.0
            for i, p in enumerate(probs):
                c += p
                if r <= c:
                    return i
            return 2

        def update(self, my_move, opp_move):
            for arr, move in ((self.my_recent, my_move), (self.opp_recent, opp_move)):
                for i in range(3):
                    arr[i] *= 0.86
                arr[move] += 1.0

            self.my_counts[my_move] += 1
            self.opp_counts[opp_move] += 1
            self.history.append((my_move, opp_move))

    def __init__(self):
        super().__init__(name="GPT")
        self.reset()

    def _seed_from_state(self, salt):
        state_repr = repr(random.getstate())
        h = 1469598103934665603 ^ salt
        for ch in state_repr[:1200]:
            h ^= ord(ch)
            h = (h * 1099511628211) & ((1 << 64) - 1)
        return h

    def reset(self):
        super().reset()
        self.rng = random.Random(self._seed_from_state(0x1234ABCD))
        self.mode_gpt = self._ModeGPTV2(self._seed_from_state(0xA1))
        self.shadow3 = self._ShadowGeminiV3()
        self.shadow4 = self._ShadowGeminiV4()
        self.shadow5 = self._ShadowGeminiV5()
        self.safe = self._ModeSafe(self._seed_from_state(0xD4))

        self.mode_scores = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.last_mode_actions = [0, 0, 0, 0, 0]
        self.recent_results = []

    def _counter(self, move):
        return (move + 2) % 3

    def _payoff(self, a, b):
        if a == b:
            return 0
        return 1 if (a - b) % 3 == 2 else -1

    def _shadow_choose(self, shadow):
        state = random.getstate()
        try:
            return shadow.choose(list(shadow.history))
        finally:
            random.setstate(state)

    def choose(self, history: list) -> int:
        a0 = self.mode_gpt.choose(list(self.mode_gpt.history))
        a1 = self._counter(self._shadow_choose(self.shadow3))
        a2 = self._counter(self._shadow_choose(self.shadow4))
        a3 = self._counter(self._shadow_choose(self.shadow5))
        a4 = self.safe.choose(list(self.safe.history))

        acts = [a0, a1, a2, a3, a4]
        self.last_mode_actions = acts

        masses = [0.0, 0.0, 0.0]
        for i, a in enumerate(acts):
            score = max(-6.0, min(6.0, self.mode_scores[i]))
            masses[a] += math.exp(0.90 * score)

        total = sum(masses)
        probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0] if total <= 0 else [m / total for m in masses]

        recent = self.recent_results[-40:]
        recent_avg = sum(recent) / len(recent) if recent else 0.0
        ordered = sorted(probs, reverse=True)
        margin = ordered[0] - ordered[1]

        n = len(self.history)
        eps = 0.05 + max(0.0, -recent_avg) * 0.12
        if n < 15:
            eps += 0.15
        elif n < 60:
            eps += 0.08
        elif n < 150:
            eps += 0.03
        if margin < 0.08:
            eps += 0.08
        elif margin < 0.15:
            eps += 0.03
        eps = max(0.03, min(0.22, eps))

        probs = [(1.0 - eps) * p + eps / 3.0 for p in probs]

        if margin > 0.18 and recent_avg > 0.04 and self.rng.random() < 0.88:
            return max(range(3), key=lambda i: probs[i])

        r = self.rng.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c:
                return i
        return 2

    def update(self, my_move: int, opp_move: int) -> None:
        coeffs = [0.30, 0.34, 0.34, 0.34, 0.20]
        for i, a in enumerate(self.last_mode_actions):
            self.mode_scores[i] = 0.92 * self.mode_scores[i] + coeffs[i] * self._payoff(a, opp_move)

        self.recent_results.append(self._payoff(my_move, opp_move))
        if len(self.recent_results) > 200:
            self.recent_results = self.recent_results[-200:]

        self.mode_gpt.update(my_move, opp_move)
        self.shadow3.update(opp_move, my_move)
        self.shadow4.update(opp_move, my_move)
        self.shadow5.update(opp_move, my_move)
        self.safe.update(my_move, opp_move)