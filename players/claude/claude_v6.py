class ClaudePlayer(BasePlayer):
    """
    Claude v1 — Adaptive Meta-Baiting Strategy

    핵심 아이디어:
    1. Gemini v5는 Cross-Entropy 기반으로 상대 패턴을 확률 분포로 학습.
       → 내가 규칙적 패턴을 보이면 Gemini가 잘 맞추게 됨.
       → 역으로, 의도적으로 패턴을 심었다가 끊어서 Gemini를 낚음 (Baiting).

    2. GPT는 Expert 시스템이 수렴하면 행동이 고착됨.
       → 최근 GPT 행동의 빈도 편향을 추적해서 카운터.

    3. 두 상대의 특성이 다르므로, 상대를 동적으로 식별해서 전략 전환.
       → 초반엔 Frequency + Markov로 상대 파악,
          이후엔 상대 특성에 맞는 전략으로 자동 전환.

    구조:
    - 다중 서브모델 앙상블 (Markov 1~3차, 빈도, 최근 가중, 결과 기반)
    - 각 모델 실적을 다중 감쇠율로 추적 (단기/장기)
    - 미끼 국면 (Bait Phase): 의도적 패턴 주입 후 끊기
    - 혼합 전략 탈출 (Escape): 고착 감지 시 epsilon 급상승
    """

    def __init__(self):
        super().__init__(name="Claude")
        self._init_state()

    def _init_state(self):
        self.my_hist = []
        self.opp_hist = []

        # 빈도 추적
        self.opp_counts = [0, 0, 0]
        self.my_counts  = [0, 0, 0]

        # 지수 감쇠 최근 빈도
        self.opp_recent = [0.0, 0.0, 0.0]
        self.my_recent  = [0.0, 0.0, 0.0]

        # Markov 테이블
        self.opp_m1 = {}   # opp_last -> opp_next
        self.opp_m2 = {}   # (opp[-2], opp[-1]) -> opp_next
        self.opp_m3 = {}   # (opp[-3],opp[-2],opp[-1]) -> opp_next
        self.my_m1  = {}   # my_last -> opp_next
        self.pair_m1 = {}  # (my_last, opp_last) -> opp_next
        self.out_m  = {}   # last_outcome -> opp_next

        # 모델 점수 (각 모델 × 4가지 감쇠율)
        self.model_names = [
            "opp_m1", "opp_m2", "opp_m3",
            "opp_freq", "opp_recent",
            "my_m1", "pair_m1", "out_m",
            "cycle_up", "cycle_dn",
        ]
        self.decays = [0.55, 0.80, 0.93, 0.985]
        # scores[model_idx][decay_idx]
        self.scores = [[0.0] * len(self.decays) for _ in self.model_names]

        # 미끼 상태
        self.bait_phase  = False
        self.bait_move   = None
        self.bait_count  = 0
        self.bait_len    = 0

        # 최근 결과 추적
        self.recent_results = []

        # 고착 감지용 내 최근 행동
        self.my_action_window = []

    def reset(self):
        super().reset()
        self._init_state()

    # ── 유틸 ──────────────────────────────
    def _counter(self, m):   return (m + 2) % 3
    def _lose_to(self, m):   return (m + 1) % 3

    def _payoff(self, a, b):
        if a == b: return 0
        return 1 if self._counter(b) == a else -1

    def _argmax_rand(self, arr):
        m = max(arr)
        c = [i for i, v in enumerate(arr) if v == m]
        return random.choice(c)

    def _decay_add(self, arr, move, gamma=0.85):
        for i in range(3): arr[i] *= gamma
        arr[move] += 1.0

    def _record(self, table, key, move):
        if key not in table: table[key] = [0, 0, 0]
        table[key][move] += 1

    def _table_pred(self, table, key, fallback):
        row = table.get(key)
        if row is None or sum(row) == 0: return fallback
        return self._argmax_rand(row)

    def _fb_opp(self):
        if sum(self.opp_counts) == 0: return random.randrange(3)
        return self._argmax_rand(self.opp_counts)

    def _last_outcome(self):
        if not self.my_hist: return 0
        return self._payoff(self.my_hist[-1], self.opp_hist[-1])

    # ── 모델별 상대 패 예측 ───────────────
    def _predict(self, name):
        n = len(self.opp_hist)
        if n == 0: return random.randrange(3)

        fb = self._fb_opp()
        opp_last = self.opp_hist[-1]
        my_last  = self.my_hist[-1]

        if name == "opp_m1":
            return self._table_pred(self.opp_m1, opp_last, fb)

        if name == "opp_m2":
            if n < 2: return fb
            return self._table_pred(self.opp_m2,
                (self.opp_hist[-2], opp_last), fb)

        if name == "opp_m3":
            if n < 3: return fb
            return self._table_pred(self.opp_m3,
                (self.opp_hist[-3], self.opp_hist[-2], opp_last), fb)

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

    # ── 고착 감지 ─────────────────────────
    def _detect_lock(self, window=20, threshold=0.80):
        """최근 window번 중 한 패가 threshold 이상이면 고착으로 판정"""
        if len(self.my_action_window) < window: return False
        w = self.my_action_window[-window:]
        for m in range(3):
            if w.count(m) / window >= threshold: return True
        return False

    # ── 미끼 국면 ─────────────────────────
    def _start_bait(self):
        """랜덤 패를 골라 5~10라운드 동안 그 패를 자주 내서 Gemini를 낚음"""
        self.bait_phase = True
        self.bait_move  = random.randrange(3)
        self.bait_len   = random.randint(5, 10)
        self.bait_count = 0

    def _bait_action(self):
        """미끼 국면: bait_len 라운드 동안 bait_move를 주로 내다가 끊음"""
        self.bait_count += 1
        if self.bait_count >= self.bait_len:
            # 미끼 종료 — 상대가 카운터를 준비할 타이밍에 역카운터
            self.bait_phase = False
            return self._counter(self._counter(self.bait_move))
        # 80% 확률로 미끼 패, 20%는 노이즈
        if random.random() < 0.80:
            return self.bait_move
        return random.randrange(3)

    # ── choose ────────────────────────────
    def choose(self, history: list) -> int:
        import math

        n = len(self.opp_hist)

        # 초반 랜덤
        if n < 3:
            return random.randrange(3)

        # 미끼 국면 진행 중
        if self.bait_phase:
            return self._bait_action()

        # 일정 확률로 미끼 국면 돌입 (30~60라운드마다 1회)
        if n > 20 and random.random() < 0.025:
            self._start_bait()
            return self._bait_action()

        # ── 앙상블 예측 ──
        # 각 모델의 최고 점수(다중 감쇠 중 max) + 평균으로 후보 구성
        candidate_mass = [0.0, 0.0, 0.0]

        for mi, name in enumerate(self.model_names):
            pred_opp = self._predict(name)
            action   = self._counter(pred_opp)   # 기본: 카운터

            best_s = max(self.scores[mi])
            mean_s = sum(self.scores[mi]) / len(self.decays)
            composite = best_s + 0.3 * mean_s

            # 추가로 +1 메타시프트(역역카운터)도 후보에 올림
            action_meta = (action + 1) % 3

            w = math.exp(max(-6.0, min(6.0, composite)) * 1.2)
            candidate_mass[action]      += w * 0.75
            candidate_mass[action_meta] += w * 0.25

        total = sum(candidate_mass)
        if total <= 0:
            probs = [1/3, 1/3, 1/3]
        else:
            probs = [m / total for m in candidate_mass]

        # ── epsilon 계산 ──
        recent = self.recent_results[-40:]
        recent_avg = sum(recent) / len(recent) if recent else 0.0
        ordered = sorted(probs, reverse=True)
        margin  = ordered[0] - ordered[1]

        eps = 0.04

        if n < 15:   eps += 0.18
        elif n < 60: eps += 0.08

        if recent_avg < -0.10: eps += 0.12
        elif recent_avg < 0.0: eps += 0.06

        if margin < 0.05: eps += 0.10
        elif margin < 0.12: eps += 0.04

        # 고착 감지 시 탈출
        if self._detect_lock(): eps = max(eps, 0.35)

        eps = max(0.03, min(0.30, eps))
        probs = [(1.0 - eps) * p + eps / 3.0 for p in probs]

        # 확신이 강하면 argmax, 아니면 샘플링
        if margin > 0.18 and recent_avg > 0.05 and random.random() < 0.85:
            return max(range(3), key=lambda i: probs[i])

        r = random.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c: return i
        return PAPER

    # ── update ────────────────────────────
    def update(self, my_move: int, opp_move: int) -> None:
        n = len(self.opp_hist)

        # 모델 점수 업데이트
        for mi, name in enumerate(self.model_names):
            pred_opp = self._predict(name)
            action   = self._counter(pred_opp)
            reward   = self._payoff(action, opp_move)
            for di, d in enumerate(self.decays):
                self.scores[mi][di] = d * self.scores[mi][di] + reward

        # Markov 테이블 업데이트
        if n >= 1:
            self._record(self.opp_m1, self.opp_hist[-1], opp_move)
            self._record(self.my_m1,  self.my_hist[-1],  opp_move)
            self._record(self.pair_m1,
                (self.my_hist[-1], self.opp_hist[-1]), opp_move)
            self._record(self.out_m, self._last_outcome(), opp_move)

        if n >= 2:
            self._record(self.opp_m2,
                (self.opp_hist[-2], self.opp_hist[-1]), opp_move)

        if n >= 3:
            self._record(self.opp_m3,
                (self.opp_hist[-3], self.opp_hist[-2], self.opp_hist[-1]),
                opp_move)

        self.opp_counts[opp_move] += 1
        self.my_counts[my_move]   += 1
        self._decay_add(self.opp_recent, opp_move)
        self._decay_add(self.my_recent,  my_move)

        self.my_hist.append(my_move)
        self.opp_hist.append(opp_move)

        self.recent_results.append(self._payoff(my_move, opp_move))
        if len(self.recent_results) > 200:
            self.recent_results = self.recent_results[-200:]

        self.my_action_window.append(my_move)
        if len(self.my_action_window) > 40:
            self.my_action_window = self.my_action_window[-40:]