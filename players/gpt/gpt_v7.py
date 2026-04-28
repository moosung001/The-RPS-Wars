class GPTPlayer(BasePlayer):
    class _ShadowGeminiV6:
        def __init__(self, py_state, np_state):
            self.np = __import__("numpy")
            self.prng = random.Random()
            self.prng.setstate(py_state)
            self.nrng = self.np.random.RandomState()
            self.nrng.set_state(np_state)
            self.reset()

        def reset(self):
            self.my_hist = []
            self.opp_hist = []

            self.max_depth = 7
            self.meta_shifts = 3
            self.num_engines = 3 * self.max_depth * self.meta_shifts + 4

            self.log_weights = self.np.zeros(self.num_engines)
            self.last_predictions = self.np.zeros(self.num_engines, dtype=int)

            self.eta = 2.0
            self.decay = 0.95

        def _find_pattern(self, seq, depth):
            if len(seq) <= depth:
                return None
            target = tuple(seq[-depth:])
            for i in range(len(seq) - depth - 1, -1, -1):
                if tuple(seq[i:i + depth]) == target:
                    return seq[i + depth]
            return None

        def _find_pair_pattern(self, my_seq, opp_seq, depth):
            if len(my_seq) <= depth:
                return None
            target_m = tuple(my_seq[-depth:])
            target_o = tuple(opp_seq[-depth:])
            for i in range(len(my_seq) - depth - 1, -1, -1):
                if tuple(my_seq[i:i + depth]) == target_m and tuple(opp_seq[i:i + depth]) == target_o:
                    return opp_seq[i + depth]
            return None

        def choose(self):
            if len(self.opp_hist) < 2:
                move = self.prng.randint(0, 2)
                self.last_predictions.fill(move)
                return int(move)

            preds = []

            for d in range(1, self.max_depth + 1):
                p = self._find_pattern(self.opp_hist, d)
                if p is None:
                    p = self.prng.randint(0, 2)
                preds.extend([(p + 2) % 3, p, (p + 1) % 3])

            for d in range(1, self.max_depth + 1):
                p = self._find_pattern(self.my_hist, d)
                if p is None:
                    p = self.prng.randint(0, 2)
                opp_guess = (p + 2) % 3
                preds.extend([(opp_guess + 2) % 3, opp_guess, (opp_guess + 1) % 3])

            for d in range(1, self.max_depth + 1):
                p = self._find_pair_pattern(self.my_hist, self.opp_hist, d)
                if p is None:
                    p = self.prng.randint(0, 2)
                preds.extend([(p + 2) % 3, p, (p + 1) % 3])

            counts = [self.opp_hist.count(0), self.opp_hist.count(1), self.opp_hist.count(2)]
            freq_p = int(self.np.argmax(counts)) if sum(counts) > 0 else self.prng.randint(0, 2)
            preds.extend([(freq_p + 2) % 3, freq_p, (freq_p + 1) % 3])

            preds.append(self.prng.randint(0, 2))
            self.last_predictions = self.np.array(preds, dtype=int)

            max_lw = self.np.max(self.log_weights)
            exp_w = self.np.exp(self.log_weights - max_lw)
            probs = exp_w / self.np.sum(exp_w)

            idx = int(self.nrng.choice(self.num_engines, p=probs))
            return int(self.last_predictions[idx])

        def update(self, my_move, opp_move):
            if self.my_hist:
                lp = self.last_predictions
                rewards = self.np.full(self.num_engines, -1.0)
                rewards[lp == opp_move] = 0.0
                rewards[lp == ((opp_move + 2) % 3)] = 1.0
                self.log_weights = self.log_weights * self.decay + self.eta * rewards

            self.my_hist.append(my_move)
            self.opp_hist.append(opp_move)

    class _ShadowClaudeV6:
        def __init__(self, py_state):
            self.prng = random.Random()
            self.prng.setstate(py_state)
            self.reset()

        def reset(self):
            self.my_hist = []
            self.opp_hist = []

            self.opp_counts = [0, 0, 0]
            self.my_counts = [0, 0, 0]

            self.opp_recent = [0.0, 0.0, 0.0]
            self.my_recent = [0.0, 0.0, 0.0]

            self.opp_m1 = {}
            self.opp_m2 = {}
            self.opp_m3 = {}
            self.my_m1 = {}
            self.pair_m1 = {}
            self.out_m = {}

            self.model_names = [
                "opp_m1", "opp_m2", "opp_m3",
                "opp_freq", "opp_recent",
                "my_m1", "pair_m1", "out_m",
                "cycle_up", "cycle_dn",
            ]
            self.decays = [0.55, 0.80, 0.93, 0.985]
            self.scores = [[0.0] * 4 for _ in self.model_names]

            self.bait_phase = False
            self.bait_move = None
            self.bait_count = 0
            self.bait_len = 0

            self.recent_results = []
            self.my_action_window = []

        def _counter(self, m):
            return (m + 2) % 3

        def _payoff(self, a, b):
            if a == b:
                return 0
            return 1 if self._counter(b) == a else -1

        def _argmax_rand(self, arr):
            m = max(arr)
            cand = [i for i, v in enumerate(arr) if v == m]
            return self.prng.choice(cand)

        def _decay_add(self, arr, move, gamma=0.85):
            for i in range(3):
                arr[i] *= gamma
            arr[move] += 1.0

        def _record(self, table, key, move):
            if key not in table:
                table[key] = [0, 0, 0]
            table[key][move] += 1

        def _table_pred(self, table, key, fallback):
            row = table.get(key)
            if row is None or sum(row) == 0:
                return fallback
            return self._argmax_rand(row)

        def _fb_opp(self):
            if sum(self.opp_counts) == 0:
                return self.prng.randrange(3)
            return self._argmax_rand(self.opp_counts)

        def _last_outcome(self):
            if not self.my_hist:
                return 0
            return self._payoff(self.my_hist[-1], self.opp_hist[-1])

        def _predict(self, name):
            n = len(self.opp_hist)
            if n == 0:
                return self.prng.randrange(3)

            fb = self._fb_opp()
            opp_last = self.opp_hist[-1]
            my_last = self.my_hist[-1]

            if name == "opp_m1":
                return self._table_pred(self.opp_m1, opp_last, fb)

            if name == "opp_m2":
                if n < 2:
                    return fb
                return self._table_pred(self.opp_m2, (self.opp_hist[-2], opp_last), fb)

            if name == "opp_m3":
                if n < 3:
                    return fb
                return self._table_pred(
                    self.opp_m3,
                    (self.opp_hist[-3], self.opp_hist[-2], opp_last),
                    fb,
                )

            if name == "opp_freq":
                return self._argmax_rand(self.opp_counts)

            if name == "opp_recent":
                return self._argmax_rand(self.opp_recent)

            if name == "my_m1":
                return self._table_pred(self.my_m1, my_last, fb)

            if name == "pair_m1":
                return self._table_pred(self.pair_m1, (my_last, opp_last), fb)

            if name == "out_m":
                return self._table_pred(self.out_m, self._last_outcome(), fb)

            if name == "cycle_up":
                return (opp_last + 1) % 3

            if name == "cycle_dn":
                return (opp_last + 2) % 3

            return fb

        def _detect_lock(self, window=20, threshold=0.80):
            if len(self.my_action_window) < window:
                return False
            w = self.my_action_window[-window:]
            for m in range(3):
                if w.count(m) / window >= threshold:
                    return True
            return False

        def _start_bait(self):
            self.bait_phase = True
            self.bait_move = self.prng.randrange(3)
            self.bait_len = self.prng.randint(5, 10)
            self.bait_count = 0

        def _bait_action(self):
            self.bait_count += 1
            if self.bait_count >= self.bait_len:
                self.bait_phase = False
                return self._counter(self._counter(self.bait_move))
            if self.prng.random() < 0.80:
                return self.bait_move
            return self.prng.randrange(3)

        def choose(self):
            n = len(self.opp_hist)

            if n < 3:
                return self.prng.randrange(3)

            if self.bait_phase:
                return self._bait_action()

            if n > 20 and self.prng.random() < 0.025:
                self._start_bait()
                return self._bait_action()

            candidate_mass = [0.0, 0.0, 0.0]

            for mi, name in enumerate(self.model_names):
                pred_opp = self._predict(name)
                action = self._counter(pred_opp)

                best_s = max(self.scores[mi])
                mean_s = sum(self.scores[mi]) / 4.0
                composite = best_s + 0.3 * mean_s

                action_meta = (action + 1) % 3
                w = math.exp(max(-6.0, min(6.0, composite)) * 1.2)
                candidate_mass[action] += w * 0.75
                candidate_mass[action_meta] += w * 0.25

            total = sum(candidate_mass)
            if total <= 0:
                probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            else:
                probs = [m / total for m in candidate_mass]

            recent = self.recent_results[-40:]
            recent_avg = sum(recent) / len(recent) if recent else 0.0
            ordered = sorted(probs, reverse=True)
            margin = ordered[0] - ordered[1]

            eps = 0.04
            if n < 15:
                eps += 0.18
            elif n < 60:
                eps += 0.08

            if recent_avg < -0.10:
                eps += 0.12
            elif recent_avg < 0.0:
                eps += 0.06

            if margin < 0.05:
                eps += 0.10
            elif margin < 0.12:
                eps += 0.04

            if self._detect_lock():
                eps = max(eps, 0.35)

            eps = max(0.03, min(0.30, eps))
            probs = [(1.0 - eps) * p + eps / 3.0 for p in probs]

            if margin > 0.18 and recent_avg > 0.05 and self.prng.random() < 0.85:
                return max(range(3), key=lambda i: probs[i])

            r = self.prng.random()
            c = 0.0
            for i, p in enumerate(probs):
                c += p
                if r <= c:
                    return i
            return 2

        def update(self, my_move, opp_move):
            n = len(self.opp_hist)

            for mi, name in enumerate(self.model_names):
                pred_opp = self._predict(name)
                action = self._counter(pred_opp)
                reward = self._payoff(action, opp_move)
                for di, d in enumerate(self.decays):
                    self.scores[mi][di] = d * self.scores[mi][di] + reward

            if n >= 1:
                self._record(self.opp_m1, self.opp_hist[-1], opp_move)
                self._record(self.my_m1, self.my_hist[-1], opp_move)
                self._record(self.pair_m1, (self.my_hist[-1], self.opp_hist[-1]), opp_move)
                self._record(self.out_m, self._last_outcome(), opp_move)

            if n >= 2:
                self._record(self.opp_m2, (self.opp_hist[-2], self.opp_hist[-1]), opp_move)

            if n >= 3:
                self._record(
                    self.opp_m3,
                    (self.opp_hist[-3], self.opp_hist[-2], self.opp_hist[-1]),
                    opp_move,
                )

            self.opp_counts[opp_move] += 1
            self.my_counts[my_move] += 1
            self._decay_add(self.opp_recent, opp_move)
            self._decay_add(self.my_recent, my_move)

            self.my_hist.append(my_move)
            self.opp_hist.append(opp_move)

            self.recent_results.append(self._payoff(my_move, opp_move))
            if len(self.recent_results) > 200:
                self.recent_results = self.recent_results[-200:]

            self.my_action_window.append(my_move)
            if len(self.my_action_window) > 40:
                self.my_action_window = self.my_action_window[-40:]

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

    def _counter(self, move):
        return (move + 2) % 3

    def reset(self):
        super().reset()

        np = __import__("numpy")
        py_state = random.getstate()
        np_state = np.random.get_state()

        self.shadow_gemini = self._ShadowGeminiV6(py_state, np_state)
        self.shadow_claude = self._ShadowClaudeV6(py_state)

        self.local_rng = random.Random(self._seed_from_state(0xC0FFEE))

        self.gemini_score = 0.0
        self.claude_score = 0.0

        self.last_gemini_pred = 0
        self.last_claude_pred = 0

        self.lock = -1

    def choose(self, history: list) -> int:
        if self.lock == 0:
            self.last_gemini_pred = self.shadow_gemini.choose()
            return self._counter(self.last_gemini_pred)

        if self.lock == 1:
            self.last_claude_pred = self.shadow_claude.choose()
            return self._counter(self.last_claude_pred)

        self.last_gemini_pred = self.shadow_gemini.choose()
        self.last_claude_pred = self.shadow_claude.choose()

        if self.last_gemini_pred == self.last_claude_pred:
            return self._counter(self.last_gemini_pred)

        if self.gemini_score > self.claude_score:
            return self._counter(self.last_gemini_pred)

        if self.claude_score > self.gemini_score:
            return self._counter(self.last_claude_pred)

        probs_opp = [0.0, 0.0, 0.0]
        probs_opp[self.last_gemini_pred] += 0.5
        probs_opp[self.last_claude_pred] += 0.5

        values = [
            probs_opp[1] - probs_opp[2],
            probs_opp[2] - probs_opp[0],
            probs_opp[0] - probs_opp[1],
        ]
        best = max(values)
        candidates = [i for i, v in enumerate(values) if v == best]
        return self.local_rng.choice(candidates)

    def update(self, my_move: int, opp_move: int) -> None:
        if self.lock == -1:
            self.gemini_score = 0.92 * self.gemini_score + (3.0 if self.last_gemini_pred == opp_move else -4.0)
            self.claude_score = 0.92 * self.claude_score + (3.0 if self.last_claude_pred == opp_move else -4.0)

            self.shadow_gemini.update(opp_move, my_move)
            self.shadow_claude.update(opp_move, my_move)

            if self.gemini_score - self.claude_score > 5.0:
                self.lock = 0
            elif self.claude_score - self.gemini_score > 5.0:
                self.lock = 1

        elif self.lock == 0:
            self.shadow_gemini.update(opp_move, my_move)

        else:
            self.shadow_claude.update(opp_move, my_move)