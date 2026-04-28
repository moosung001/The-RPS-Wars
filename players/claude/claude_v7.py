import random
import math
import numpy as np

class ClaudePlayer(BasePlayer):
    """
    Claude v7 — Triple Shadow Counter + CE Ensemble

    전략:
    1. GPT v6의 핵심 모드(ModeGPTV2)를 내부 시뮬레이션 → 카운터
    2. Gemini v6의 Hedge 앙상블을 내부 시뮬레이션 → 카운터
    3. 자체 Cross-Entropy 기반 앙상블 (Gemini v5 스타일, 검증된 강함)
    4. 메타 점수로 세 모드 실시간 선택
    5. shadow 호출 시 random.getstate()/setstate()로 전역 RNG 격리
    """

    # ──────────────────────────────────────────
    # Shadow GPT (ModeGPTV2 근사)
    # ──────────────────────────────────────────
    class _ShadowGPT:
        def __init__(self):
            self.rng = random.Random(0xDEADBEEF)
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
            base_names = [
                "opp_last", "opp_freq", "opp_recent", "opp_m1", "opp_m2",
                "pair", "out_opp", "cycle_up", "cycle_down",
                "my_last", "my_freq", "my_recent", "my_m1", "my_m2", "out_my",
            ]
            self.experts = [(n, r) for n in base_names for r in (0, 1, 2)]
            self.logw = [0.0] * len(self.experts)
            self.last_actions = [0] * len(self.experts)
            self.recent_results = []

        def _pay(self, a, b):
            if a == b: return 0
            return 1 if (a - b) % 3 == 2 else -1

        def _da(self, arr, move, g=0.85):
            for i in range(3): arr[i] *= g
            arr[move] += 1.0

        def _amr(self, xs):
            m = max(xs); c = [i for i, x in enumerate(xs) if x == m]
            return self.rng.choice(c)

        def _tp(self, table, key, fb):
            row = table.get(key)
            return fb if row is None else self._amr(row)

        def _lo(self):
            if not self.my_hist: return 0
            return self._pay(self.my_hist[-1], self.opp_hist[-1])

        def _sym(self, name):
            n = len(self.opp_hist)
            if n == 0: return self.rng.randrange(3)
            ol = self.opp_hist[-1]; ml = self.my_hist[-1]
            fo = self._amr(self.opp_counts)
            fm = self._amr(self.my_counts) if sum(self.my_counts) > 0 else self.rng.randrange(3)
            if name == "opp_last": return ol
            if name == "opp_freq": return self._amr(self.opp_counts)
            if name == "opp_recent": return self._amr(self.opp_recent)
            if name == "opp_m1": return self._tp(self.opp_m1, ol, fo)
            if name == "opp_m2":
                if n < 2: return fo
                return self._tp(self.opp_m2, (self.opp_hist[-2], ol), fo)
            if name == "pair": return self._tp(self.pair, (ml, ol), fo)
            if name == "out_opp": return self._tp(self.out_opp, self._lo(), fo)
            if name == "cycle_up": return (ol + 1) % 3
            if name == "cycle_down": return (ol + 2) % 3
            if name == "my_last": return ml
            if name == "my_freq": return self._amr(self.my_counts)
            if name == "my_recent": return self._amr(self.my_recent)
            if name == "my_m1": return self._tp(self.my_m1, ml, fm)
            if name == "my_m2":
                if n < 2: return fm
                return self._tp(self.my_m2, (self.my_hist[-2], ml), fm)
            if name == "out_my": return self._tp(self.out_my, self._lo(), fm)
            return self.rng.randrange(3)

        def choose(self, history):
            masses = [0.0, 0.0, 0.0]
            acts = []
            for idx, (name, rot) in enumerate(self.experts):
                x = self._sym(name)
                a = (x + rot) % 3
                acts.append(a)
                lw = max(-8.0, min(8.0, self.logw[idx]))
                masses[a] += math.exp(lw)
            self.last_actions = acts
            total = sum(masses)
            probs = [1/3, 1/3, 1/3] if total <= 0 else [m/total for m in masses]
            recent = self.recent_results[-30:]
            ra = sum(recent)/len(recent) if recent else 0.0
            spread = max(probs) - min(probs)
            eps = 0.06 + max(0.0, -ra) * 0.45
            n = len(self.opp_hist)
            if n < 15: eps += 0.18
            elif n < 60: eps += 0.10
            elif n < 150: eps += 0.05
            if spread < 0.12: eps += 0.08
            eps = max(0.08, min(0.35, eps))
            probs = [(1-eps)*p + eps/3.0 for p in probs]
            r = self.rng.random(); c = 0.0
            for i, p in enumerate(probs):
                c += p
                if r <= c: return i
            return 2

        def update(self, my_move, opp_move):
            for i, a in enumerate(self.last_actions):
                self.logw[i] = 0.97 * self.logw[i] + 0.22 * self._pay(a, opp_move)
            n = len(self.opp_hist)
            if n >= 1:
                self.opp_m1.setdefault(self.opp_hist[-1], [0,0,0])[opp_move] += 1
                self.my_m1.setdefault(self.my_hist[-1], [0,0,0])[my_move] += 1
                self.pair.setdefault((self.my_hist[-1], self.opp_hist[-1]), [0,0,0])[opp_move] += 1
                prev = self._lo()
                self.out_opp.setdefault(prev, [0,0,0])[opp_move] += 1
                self.out_my.setdefault(prev, [0,0,0])[my_move] += 1
            if n >= 2:
                self.opp_m2.setdefault((self.opp_hist[-2], self.opp_hist[-1]), [0,0,0])[opp_move] += 1
                self.my_m2.setdefault((self.my_hist[-2], self.my_hist[-1]), [0,0,0])[my_move] += 1
            self.opp_counts[opp_move] += 1
            self.my_counts[my_move] += 1
            self._da(self.opp_recent, opp_move)
            self._da(self.my_recent, my_move)
            self.my_hist.append(my_move)
            self.opp_hist.append(opp_move)
            self.history.append((my_move, opp_move))
            self.recent_results.append(self._pay(my_move, opp_move))
            if len(self.recent_results) > 200:
                self.recent_results = self.recent_results[-200:]

    # ──────────────────────────────────────────
    # Shadow Gemini v6 (Hedge 앙상블 근사)
    # ──────────────────────────────────────────
    class _ShadowGemini:
        def __init__(self):
            self.reset()

        def reset(self):
            self.history = []
            self.my_hist = []
            self.opp_hist = []
            self.max_depth = 7
            self.num_engines = 3 * 7 * 3 + 4  # 67
            self.log_weights = np.zeros(self.num_engines)
            self.last_predictions = np.zeros(self.num_engines, dtype=int)
            self.eta = 2.0
            self.decay = 0.95

        def _find_pattern(self, seq, depth):
            if len(seq) <= depth: return None
            target = tuple(seq[-depth:])
            for i in range(len(seq) - depth - 1, -1, -1):
                if tuple(seq[i:i+depth]) == target:
                    return seq[i+depth]
            return None

        def _find_pair_pattern(self, my_seq, opp_seq, depth):
            if len(my_seq) <= depth: return None
            tm = tuple(my_seq[-depth:]); to = tuple(opp_seq[-depth:])
            for i in range(len(my_seq) - depth - 1, -1, -1):
                if tuple(my_seq[i:i+depth]) == tm and tuple(opp_seq[i:i+depth]) == to:
                    return opp_seq[i+depth]
            return None

        def choose(self, history):
            if len(self.opp_hist) < 2:
                move = random.randint(0, 2)
                self.last_predictions.fill(move)
                return move

            preds = []
            for d in range(1, self.max_depth + 1):
                p = self._find_pattern(self.opp_hist, d)
                if p is None: p = random.randint(0, 2)
                preds.extend([(p+2)%3, p, (p+1)%3])
            for d in range(1, self.max_depth + 1):
                p = self._find_pattern(self.my_hist, d)
                if p is None: p = random.randint(0, 2)
                og = (p+2)%3
                preds.extend([(og+2)%3, og, (og+1)%3])
            for d in range(1, self.max_depth + 1):
                p = self._find_pair_pattern(self.my_hist, self.opp_hist, d)
                if p is None: p = random.randint(0, 2)
                preds.extend([(p+2)%3, p, (p+1)%3])
            counts = [self.opp_hist.count(0), self.opp_hist.count(1), self.opp_hist.count(2)]
            fp = int(np.argmax(counts)) if sum(counts) > 0 else random.randint(0, 2)
            preds.extend([(fp+2)%3, fp, (fp+1)%3])
            preds.append(random.randint(0, 2))

            self.last_predictions = np.array(preds)
            mw = np.max(self.log_weights)
            ew = np.exp(self.log_weights - mw)
            probs = ew / np.sum(ew)
            idx = int(np.random.choice(self.num_engines, p=probs))
            return int(self.last_predictions[idx])

        def update(self, my_move, opp_move):
            if self.my_hist:
                rewards = np.zeros(self.num_engines)
                for i, p in enumerate(self.last_predictions):
                    if p == (opp_move + 2) % 3: rewards[i] = 1.0
                    elif p == opp_move: rewards[i] = 0.0
                    else: rewards[i] = -1.0
                self.log_weights = self.log_weights * self.decay + self.eta * rewards
            self.my_hist.append(my_move)
            self.opp_hist.append(opp_move)
            self.history.append((my_move, opp_move))

    # ──────────────────────────────────────────
    # 자체 CE 앙상블 (Gemini v5 스타일)
    # ──────────────────────────────────────────
    class _OwnEngine:
        def __init__(self):
            self.reset()

        def reset(self):
            self.history = []
            self.my_history = []
            self.op_history = []
            self.max_depth = 6
            self.num_engines = 3 * 6 * 3 + 3 + 1  # 58
            self.weights = np.zeros(self.num_engines)
            self.last_probs = np.ones((self.num_engines, 3)) / 3.0
            self.lr = 0.5
            self.decay = 0.98

        def get_dist(self, seq, depth):
            dist = np.ones(3) * 0.1
            if len(seq) <= depth: return dist
            target = tuple(seq[-depth:])
            for i in range(len(seq) - depth - 1, -1, -1):
                if tuple(seq[i:i+depth]) == target:
                    dist[self.op_history[i+depth]] += 1.0
            return dist

        def get_comb_dist(self, comb, depth):
            dist = np.ones(3) * 0.1
            if len(comb) <= depth: return dist
            target = tuple(comb[-depth:])
            for i in range(len(comb) - depth - 1, -1, -1):
                if tuple(comb[i:i+depth]) == target:
                    dist[self.op_history[i+depth]] += 1.0
            return dist

        def choose(self, history):
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
            all_probs.append(np.array([1/3, 1/3, 1/3]))
            self.last_probs = np.array(all_probs)
            ew = np.exp(self.weights - np.max(self.weights))
            mp = ew / np.sum(ew)
            final = np.sum(self.last_probs * mp[:, None], axis=0)
            return int((int(np.argmax(final)) - 1) % 3)

        def update(self, my_move, opp_move):
            if self.my_history:
                ap = self.last_probs[:, opp_move]
                ap = np.clip(ap, 1e-7, 1.0)
                self.weights = self.weights * self.decay + self.lr * np.log(ap)
            self.my_history.append(my_move)
            self.op_history.append(opp_move)
            self.history.append((my_move, opp_move))

    # ──────────────────────────────────────────
    # ClaudePlayer 본체
    # ──────────────────────────────────────────
    def __init__(self):
        super().__init__(name="Claude")
        self.reset()

    def reset(self):
        super().reset()
        self.shadow_gpt    = self._ShadowGPT()
        self.shadow_gemini = self._ShadowGemini()
        self.own_engine    = self._OwnEngine()

        # 모드 점수: [counter_gpt, counter_gemini, own_ce]
        self.mode_scores = [0.0, 0.0, 0.0]
        self.last_mode_actions = [0, 0, 0]
        self.recent_results = []
        self.rng = random.Random(0xC1A0DE7)

    def _counter(self, m): return (m + 2) % 3
    def _payoff(self, a, b):
        if a == b: return 0
        return 1 if self._counter(b) == a else -1

    def _safe_choose(self, shadow):
        """전역 RNG 오염 없이 shadow 호출"""
        state = random.getstate()
        np_state = np.random.get_state()
        try:
            return shadow.choose(list(shadow.history))
        finally:
            random.setstate(state)
            np.random.set_state(np_state)

    def choose(self, history: list) -> int:
        # 각 모드의 액션 계산
        gpt_pred    = self._safe_choose(self.shadow_gpt)
        gemini_pred = self._safe_choose(self.shadow_gemini)
        own_act     = self._safe_choose(self.own_engine)

        a0 = self._counter(gpt_pred)    # GPT가 낼 패를 카운터
        a1 = self._counter(gemini_pred) # Gemini가 낼 패를 카운터
        a2 = own_act                    # 자체 CE 앙상블

        acts = [a0, a1, a2]
        self.last_mode_actions = acts

        # 메타 점수 기반 가중합
        masses = [0.0, 0.0, 0.0]
        for i, a in enumerate(acts):
            score = max(-6.0, min(6.0, self.mode_scores[i]))
            masses[a] += math.exp(0.90 * score)

        total = sum(masses)
        probs = [1/3, 1/3, 1/3] if total <= 0 else [m/total for m in masses]

        recent = self.recent_results[-40:]
        recent_avg = sum(recent)/len(recent) if recent else 0.0
        ordered = sorted(probs, reverse=True)
        margin = ordered[0] - ordered[1]

        n = len(self.history)
        eps = 0.04 + max(0.0, -recent_avg) * 0.10
        if n < 15:   eps += 0.15
        elif n < 60: eps += 0.08
        elif n < 150: eps += 0.03
        if margin < 0.08:  eps += 0.08
        elif margin < 0.15: eps += 0.03
        eps = max(0.03, min(0.22, eps))

        probs = [(1-eps)*p + eps/3.0 for p in probs]

        if margin > 0.18 and recent_avg > 0.04 and self.rng.random() < 0.88:
            return max(range(3), key=lambda i: probs[i])

        r = self.rng.random(); c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c: return i
        return 2

    def update(self, my_move: int, opp_move: int) -> None:
        # 모드 점수 업데이트
        for i, a in enumerate(self.last_mode_actions):
            self.mode_scores[i] = 0.92 * self.mode_scores[i] + 0.33 * self._payoff(a, opp_move)

        self.recent_results.append(self._payoff(my_move, opp_move))
        if len(self.recent_results) > 200:
            self.recent_results = self.recent_results[-200:]

        # shadow 및 자체 엔진 업데이트 (실제 패로)
        self.shadow_gpt.update(my_move, opp_move)    # GPT처럼 행동했다면
        self.shadow_gemini.update(my_move, opp_move) # Gemini처럼 행동했다면
        self.own_engine.update(my_move, opp_move)