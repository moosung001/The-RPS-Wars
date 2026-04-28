"""
가위바위보 AI 리그전 프레임워크
================================
심판: Claude
선수: GPT / Gemini / Claude

매치업: GPT vs Gemini / GPT vs Claude / Gemini vs Claude
방식: 각 매치업을 N_TRIALS회 반복, 평균 승률로 승점 산정
승점: 승리 3점 / 통계적 무승부 1점 / 패배 0점

실행: python rps_league.py
"""

import random
import statistics
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from abc import ABC, abstractmethod

# ──────────────────────────────────────────
# 0. 공통 상수
# ──────────────────────────────────────────
ROCK, SCISSORS, PAPER = 0, 1, 2

def beats(a, b):
    return (a == ROCK and b == SCISSORS) or \
           (a == SCISSORS and b == PAPER) or \
           (a == PAPER and b == ROCK)

# ──────────────────────────────────────────
# 1. 베이스 클래스 (수정 금지)
# ──────────────────────────────────────────
class BasePlayer(ABC):
    def __init__(self, name: str):
        self.name = name
        self.history = []

    @abstractmethod
    def choose(self, history: list) -> int:
        pass

    @abstractmethod
    def update(self, my_move: int, opp_move: int) -> None:
        pass

    def reset(self):
        self.history = []


# ──────────────────────────────────────────
# 2. GPT 슬롯  ← GPT 코드를 여기에 붙여넣으세요
# ──────────────────────────────────────────
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
# ──────────────────────────────────────────
# 3. Gemini 슬롯  ← Gemini 코드를 여기에 붙여넣으세요
# ──────────────────────────────────────────
import time
import math
import numpy as np

class GeminiPlayer(BasePlayer):
    """
    7세대 최종 진화형: Decoupled RNG 기반 EMA 텐서 MAB (Multi-Armed Bandit)
    
    1. GPT_v6의 Shadow 모델 및 RNG 동기화를 완벽히 무력화하기 위한 독립적인 Local 난수 생성기 사용.
    2. 지수이동평균(EMA) 정규화를 도입하여 극단적으로 다른 5개의 감쇠율(Decay)을 완벽히 조율.
    3. 27개의 컨텍스트 트리 × 3 메타 시프트 × 5 감쇠율 = 405개의 동적 전문가 엔진을 
       Temperature 20.0의 Softmax 혼합 전략(Mixed Strategy)으로 배합하여 내시 균형 달성.
    """
    def __init__(self):
        super().__init__(name="Gemini")
        # Framework의 random.seed()에 영향을 받지 않는 완벽히 독립된 고유 난수 생성기
        seed = (int(time.time() * 1000) ^ id(self)) % (2**32)
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        super().reset()
        self.my_hist = []
        self.opp_hist = []
        self.pair_hist = []
        self.out_hist = []

        self.num_base = 27
        self.meta_shifts = 3
        self.num_actions = self.num_base * self.meta_shifts
        self.decays = np.array([0.60, 0.80, 0.90, 0.97, 0.99])
        self.num_decays = len(self.decays)

        # EMA 정규화를 위한 스코어 매트릭스: Shape (81, 5)
        self.scores = np.zeros((self.num_actions, self.num_decays))
        self.last_actions = np.zeros(self.num_actions, dtype=int)

        # 고속 컨텍스트 탐색 트리 (N-gram Dictionary)
        self.ctx_opp = [{} for _ in range(7)]
        self.ctx_my = [{} for _ in range(7)]
        self.ctx_pair = [{} for _ in range(7)]
        self.ctx_out = [{} for _ in range(7)]

        self.opp_counts = np.zeros(3)
        self.my_counts = np.zeros(3)
        self.opp_recent = np.zeros(3)
        self.my_recent = np.zeros(3)

    def _update_tree(self, tree, hist, move):
        """다중 깊이 컨텍스트 트리에 새로운 결과 학습"""
        for d in range(1, 7):
            if len(hist) >= d:
                ctx = tuple(hist[-d:])
                if ctx not in tree[d]:
                    tree[d][ctx] = np.zeros(3)
                tree[d][ctx][move] += 1.0

    def _predict_tree_depth(self, tree, hist, d, fallback):
        """특정 깊이의 컨텍스트를 기반으로 상대 패 예측"""
        if len(hist) >= d:
            ctx = tuple(hist[-d:])
            if ctx in tree[d]:
                counts = tree[d][ctx]
                max_val = np.max(counts)
                cands = np.where(counts == max_val)[0]
                return self.rng.choice(cands)
        return fallback

    def _predict_tree_longest(self, tree, hist, fallback):
        """가장 깊은 일치 패턴을 찾아 상대 패 예측"""
        for d in range(6, 0, -1):
            if len(hist) >= d:
                ctx = tuple(hist[-d:])
                if ctx in tree[d]:
                    counts = tree[d][ctx]
                    max_val = np.max(counts)
                    cands = np.where(counts == max_val)[0]
                    return self.rng.choice(cands)
        return fallback

    def choose(self, history: list) -> int:
        if len(self.opp_hist) == 0:
            return self.rng.randint(3)

        # --- 기본 베이스라인 추론 ---
        if np.sum(self.opp_counts) > 0:
            fb_opp = self.rng.choice(np.where(self.opp_counts == np.max(self.opp_counts))[0])
        else:
            fb_opp = self.rng.randint(3)

        if np.sum(self.my_counts) > 0:
            fb_my = self.rng.choice(np.where(self.my_counts == np.max(self.my_counts))[0])
        else:
            fb_my = self.rng.randint(3)

        preds = np.zeros(self.num_base, dtype=int)
        
        # 1. 0~5: 상대 패 히스토리 기반 (깊이 1~6)
        for d in range(1, 7):
            preds[d-1] = self._predict_tree_depth(self.ctx_opp, self.opp_hist, d, fb_opp)
        
        # 2. 6~11: 내 패 히스토리 기반 (상대가 내 패턴을 읽는다고 역가정)
        for d in range(1, 7):
            preds[5+d] = self._predict_tree_depth(self.ctx_my, self.my_hist, d, fb_opp)
            
        # 3. 12~17: 양측 교차 복합 패턴
        for d in range(1, 7):
            preds[11+d] = self._predict_tree_depth(self.ctx_pair, self.pair_hist, d, fb_opp)
            
        # 4. 18~21: 단순 빈도 및 단기 편향 분석
        preds[18] = fb_opp
        preds[19] = (fb_my + 2) % 3  # 상대가 내가 가장 많이 내는 패를 잡으려 할 때
        preds[20] = np.argmax(self.opp_recent) if np.sum(self.opp_recent) > 0 else fb_opp
        
        m_rec = np.argmax(self.my_recent) if np.sum(self.my_recent) > 0 else fb_my
        preds[21] = (m_rec + 2) % 3
        
        # 5. 22~26: 게임 결과 마르코프 및 행동 휴리스틱
        preds[22] = self._predict_tree_longest(self.ctx_out, self.out_hist, fb_opp)
        preds[23] = (self.opp_hist[-1] + 1) % 3  # Cycle Up
        preds[24] = (self.opp_hist[-1] + 2) % 3  # Cycle Down
        preds[25] = self.my_hist[-1]             # Copy Me
        preds[26] = (self.my_hist[-1] + 2) % 3   # Beat Me

        # --- 81개의 메타 시프트 행동 생성 ---
        actions = np.zeros(self.num_actions, dtype=int)
        for i in range(self.num_base):
            p = preds[i]
            actions[i*3 + 0] = (p + 2) % 3    # 메타 0: 직접 카운터
            actions[i*3 + 1] = p              # 메타 1: 상대의 카운터를 역카운터
            actions[i*3 + 2] = (p + 1) % 3    # 메타 2: 역카운터의 역카운터
            
        self.last_actions = actions

        # --- Softmax 확률적 내시 균형(Nash Equilibrium) 선택 ---
        flat_scores = self.scores.flatten()
        # 점수가 가장 높은 상위 15개 전문가만 필터링 (가비지 노이즈 제거)
        best_idx = np.argsort(flat_scores)[-15:] 
        
        top_actions = self.last_actions[best_idx // self.num_decays]
        top_scores = flat_scores[best_idx]
        
        # 텐서 혼합을 위한 Softmax 적용 (Temperature=20.0으로 확신도 펌핑)
        temperature = 20.0
        exp_scores = np.exp((top_scores - np.max(top_scores)) * temperature)
        probs = exp_scores / np.sum(exp_scores)
        
        action_probs = np.zeros(3)
        for a, p in zip(top_actions, probs):
            action_probs[a] += p
            
        # 안전망(Epsilon): 최근 성적이 나쁘면 무작위성(탐색) 증가
        recent_perf = np.mean(self.out_hist[-40:]) if len(self.out_hist) >= 40 else 0.0
        eps = 0.04
        if recent_perf < 0:
            eps += 0.15
        elif recent_perf < 0.1:
            eps += 0.06
            
        action_probs = (1 - eps) * action_probs + eps / 3.0
        
        final_action = self.rng.choice(3, p=action_probs)
        return int(final_action)

    def update(self, my_move: int, opp_move: int) -> None:
        if len(self.opp_hist) > 0:
            # 1. 81개 행동의 승무패 판별 벡터화 연산
            payoffs = np.zeros(self.num_actions)
            wins = (self.last_actions == (opp_move + 2) % 3)
            losses = (self.last_actions == (opp_move + 1) % 3)
            payoffs[wins] = 1.0
            payoffs[losses] = -1.0
            
            # 2. EMA(지수이동평균) 정규화 기반의 MAB 스코어 업데이트
            # (1.0 - decay)를 곱해 모든 노후화 모델의 점수 스케일을 완벽히 일치시킴
            self.scores = self.scores * self.decays + payoffs[:, np.newaxis] * (1.0 - self.decays)

        # 3. 내부 트래커 업데이트
        self._update_tree(self.ctx_opp, self.opp_hist, opp_move)
        self._update_tree(self.ctx_my, self.my_hist, opp_move)
        self._update_tree(self.ctx_pair, self.pair_hist, opp_move)
        
        # 결과 매핑: 무승부=0, 승리=1, 패배=2
        outcome = 0 if my_move == opp_move else (1 if my_move == (opp_move + 2) % 3 else 2)
        self._update_tree(self.ctx_out, self.out_hist, opp_move)

        self.opp_counts[opp_move] += 1
        self.my_counts[my_move] += 1
        
        self.opp_recent = self.opp_recent * 0.85
        self.opp_recent[opp_move] += 1.0
        self.my_recent = self.my_recent * 0.85
        self.my_recent[my_move] += 1.0

        self.opp_hist.append(opp_move)
        self.my_hist.append(my_move)
        self.pair_hist.append(my_move * 3 + opp_move)
        self.out_hist.append(outcome)
# ──────────────────────────────────────────
# 4. Claude 슬롯  ← Claude 코드를 여기에 붙여넣으세요
# ──────────────────────────────────────────
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

# ──────────────────────────────────────────
# 5. 단일 매치 엔진 (수정 금지)
# ──────────────────────────────────────────
def run_match(player_a: BasePlayer, player_b: BasePlayer, n_rounds: int):
    player_a.reset()
    player_b.reset()

    wins_a = wins_b = draws = 0
    wr_a, wr_b, wr_d = [], [], []

    for r in range(1, n_rounds + 1):
        move_a = player_a.choose(list(player_a.history))
        move_b = player_b.choose(list(player_b.history))

        assert move_a in (ROCK, SCISSORS, PAPER), f"{player_a.name}: 잘못된 패 {move_a}"
        assert move_b in (ROCK, SCISSORS, PAPER), f"{player_b.name}: 잘못된 패 {move_b}"

        if beats(move_a, move_b):
            wins_a += 1
        elif beats(move_b, move_a):
            wins_b += 1
        else:
            draws += 1

        player_a.update(move_a, move_b)
        player_b.update(move_b, move_a)

        player_a.history.append((move_a, move_b))
        player_b.history.append((move_b, move_a))

        wr_a.append(wins_a / r)
        wr_b.append(wins_b / r)
        wr_d.append(draws / r)

    return wr_a, wr_b, wr_d


# ──────────────────────────────────────────
# 6. 다회 매치업 실행 (수정 금지)
# ──────────────────────────────────────────
def run_matchup(player_a_cls, player_b_cls, n_rounds: int, n_trials: int):
    """
    두 플레이어 클래스를 n_trials번 붙여서 평균 승률과 최종 승률 리스트를 반환.
    반환: (avg_wr_a, avg_wr_b, avg_wr_d, final_a_list, final_b_list, name_a, name_b)
    """
    sum_a = [0.0] * n_rounds
    sum_b = [0.0] * n_rounds
    sum_d = [0.0] * n_rounds
    final_a, final_b = [], []

    pa_name = player_a_cls().name
    pb_name = player_b_cls().name

    print(f"\n  [{pa_name} vs {pb_name}]")
    for trial in range(n_trials):
        random.seed(42 + trial)
        pa = player_a_cls()
        pb = player_b_cls()
        wr_a, wr_b, wr_d = run_match(pa, pb, n_rounds)

        for r in range(n_rounds):
            sum_a[r] += wr_a[r]
            sum_b[r] += wr_b[r]
            sum_d[r] += wr_d[r]

        final_a.append(wr_a[-1])
        final_b.append(wr_b[-1])

        print(f"    Trial {trial+1:>3d}/{n_trials}  |  "
              f"{pa_name} {wr_a[-1]*100:.1f}%  {pb_name} {wr_b[-1]*100:.1f}%")

    avg_a = [x / n_trials for x in sum_a]
    avg_b = [x / n_trials for x in sum_b]
    avg_d = [x / n_trials for x in sum_d]

    return avg_a, avg_b, avg_d, final_a, final_b, pa_name, pb_name


# ──────────────────────────────────────────
# 7. 승점 계산 (수정 금지)
# ──────────────────────────────────────────
DRAW_THRESHOLD = 0.005   # 평균 승률 차이가 이 이하면 무승부로 판정

def calc_points(final_a, final_b, name_a, name_b):
    """
    n_trials번의 최종 승률 리스트를 받아 승점을 반환.
    통계적 무승부 판정: 평균 승률 차이 < DRAW_THRESHOLD
    반환: (points_a, points_b, result_str)
    """
    mean_a = statistics.mean(final_a)
    mean_b = statistics.mean(final_b)
    wins_a = sum(1 for a, b in zip(final_a, final_b) if a > b)
    wins_b = len(final_a) - wins_a

    diff = abs(mean_a - mean_b)

    if diff < DRAW_THRESHOLD:
        pts_a, pts_b = 1, 1
        result = f"Draw ({name_a} {mean_a*100:.2f}% vs {name_b} {mean_b*100:.2f}%)"
    elif mean_a > mean_b:
        pts_a, pts_b = 3, 0
        result = f"{name_a} Win ({mean_a*100:.2f}% vs {mean_b*100:.2f}%, {wins_a}:{wins_b})"
    else:
        pts_a, pts_b = 0, 3
        result = f"{name_b} Win ({mean_b*100:.2f}% vs {mean_a*100:.2f}%, {wins_b}:{wins_a})"

    return pts_a, pts_b, result, mean_a, mean_b


# ──────────────────────────────────────────
# 8. 시각화 (수정 금지)
# ──────────────────────────────────────────
def plot_league(matchup_results, standings, n_rounds, n_trials):
    n_matchups = len(matchup_results)
    fig = plt.figure(figsize=(14, 4 * n_matchups + 3))
    gs = fig.add_gridspec(n_matchups + 1, 1,
                          height_ratios=[3] * n_matchups + [2],
                          hspace=0.5)

    colors = {"GPT": "#10a37f", "Gemini": "#4285f4", "Claude": "#cc785c"}
    rounds = list(range(1, n_rounds + 1))

    for i, (avg_a, avg_b, avg_d, name_a, name_b, mean_a, mean_b) in enumerate(matchup_results):
        ax = fig.add_subplot(gs[i])
        ax.plot(rounds, [w * 100 for w in avg_a],
                label=name_a, color=colors.get(name_a, "#888888"), linewidth=1.8)
        ax.plot(rounds, [w * 100 for w in avg_b],
                label=name_b, color=colors.get(name_b, "#aaaaaa"), linewidth=1.8)
        ax.plot(rounds, [w * 100 for w in avg_d],
                label="Draw", color="#cccccc", linewidth=1.0, linestyle="--")
        ax.axhline(33.33, color="black", linestyle=":", linewidth=0.8,
                   alpha=0.4, label="Random baseline")
        ax.set_title(f"{name_a} vs {name_b}  —  "
                     f"{name_a} {mean_a*100:.1f}% / {name_b} {mean_b*100:.1f}%",
                     fontsize=11)
        ax.set_ylabel("Win Rate (%)")
        ax.set_ylim(0, 70)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Standings table
    ax_table = fig.add_subplot(gs[n_matchups])
    ax_table.axis("off")
    table_data = [[f"{i+1}", name, f"{pts}pts", f"{wr*100:.2f}%"]
                  for i, (name, pts, wr) in enumerate(standings)]
    col_labels = ["Rank", "Player", "Points", "Avg Win Rate"]
    tbl = ax_table.table(cellText=table_data, colLabels=col_labels,
                         loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 2.0)

    for j, (name, pts, wr) in enumerate(standings):
        c = colors.get(name, "#eeeeee")
        for k in range(4):
            tbl[(j + 1, k)].set_facecolor(c + "44")

    fig.suptitle(
        f"Rock Paper Scissors League\n"
        f"({n_rounds} Rounds × {n_trials} Trials per Matchup)",
        fontsize=14, y=1.01
    )

    plt.savefig("rps_league_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nResult saved: rps_league_result.png")

# ──────────────────────────────────────────
# 9. 메인 실행
# ──────────────────────────────────────────
if __name__ == "__main__":
    N_ROUNDS = 1000
    N_TRIALS = 30

    MATCHUPS = [
        (GPTPlayer,    GeminiPlayer),
        (GPTPlayer,    ClaudePlayer),
        (GeminiPlayer, ClaudePlayer),
    ]

    points = {GPTPlayer().name: 0,
              GeminiPlayer().name: 0,
              ClaudePlayer().name: 0}
    total_wr = {name: [] for name in points}

    print(f"[Referee: Claude] League Start!")
    print(f"  {N_ROUNDS} rounds x {N_TRIALS} trials per matchup\n")

    matchup_results = []
    match_log = []

    for cls_a, cls_b in MATCHUPS:
        avg_a, avg_b, avg_d, fa, fb, na, nb = run_matchup(
            cls_a, cls_b, N_ROUNDS, N_TRIALS
        )
        pts_a, pts_b, result_str, mean_a, mean_b = calc_points(fa, fb, na, nb)

        points[na] += pts_a
        points[nb] += pts_b
        total_wr[na].extend(fa)
        total_wr[nb].extend(fb)

        matchup_results.append((avg_a, avg_b, avg_d, na, nb, mean_a, mean_b))
        match_log.append((na, nb, result_str, pts_a, pts_b))

    standings = sorted(
        [(name, pts, statistics.mean(total_wr[name])) for name, pts in points.items()],
        key=lambda x: (x[1], x[2]),
        reverse=True
    )

    print("\n" + "=" * 55)
    print("  Matchup Results")
    print("=" * 55)
    for na, nb, result_str, pts_a, pts_b in match_log:
        print(f"  {na} vs {nb}: {result_str}")
        print(f"    -> {na} {pts_a}pts / {nb} {pts_b}pts")

    print("\n" + "=" * 55)
    print("  Final Standings")
    print("=" * 55)
    for i, (name, pts, wr) in enumerate(standings):
        print(f"  {i+1}st  {name:10s}  {pts}pts  (avg win rate {wr*100:.2f}%)")

    print(f"\n Trophy Final Winner: {standings[0][0]}")

    plot_league(matchup_results, standings, N_ROUNDS, N_TRIALS)