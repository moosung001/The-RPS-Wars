"""
가위바위보 AI 대결 프레임워크
================================
심판: Claude
선수 A: GPT   → gpt_algorithm.py 에서 GPTPlayer 구현
선수 B: Gemini → gemini_algorithm.py 에서 GeminiPlayer 구현

실행: python rps_framework.py
"""

import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from abc import ABC, abstractmethod

# ──────────────────────────────────────────
# 0. 공통 상수
# ──────────────────────────────────────────
ROCK, SCISSORS, PAPER = 0, 1, 2
MOVE_NAME = {ROCK: "바위", SCISSORS: "가위", PAPER: "보"}

def beats(a, b):
    """a가 b를 이기면 True"""
    return (a == ROCK and b == SCISSORS) or \
           (a == SCISSORS and b == PAPER) or \
           (a == PAPER and b == ROCK)

# ──────────────────────────────────────────
# 1. 플레이어 베이스 클래스 (수정 금지)
# ──────────────────────────────────────────
class BasePlayer(ABC):
    """
    GPT와 Gemini는 이 클래스를 상속받아 구현합니다.

    필수 구현 메서드:
        choose(history) -> int  : ROCK(0) / SCISSORS(1) / PAPER(2) 반환
        update(my_move, opp_move) -> None : 라운드 결과를 받아 내부 상태 업데이트

    history 형식:
        [(my_move_r1, opp_move_r1), (my_move_r2, opp_move_r2), ...]
        첫 라운드에는 빈 리스트 [] 가 전달됩니다.
    """

    def __init__(self, name: str):
        self.name = name
        self.history = []  # 프레임워크가 자동 관리

    @abstractmethod
    def choose(self, history: list) -> int:
        """다음에 낼 패를 반환. history는 (내 패, 상대 패) 튜플의 리스트."""
        pass

    @abstractmethod
    def update(self, my_move: int, opp_move: int) -> None:
        """라운드 종료 후 호출됨. 학습/상태 업데이트에 사용."""
        pass

    def reset(self):
        """새 매치 시작 시 프레임워크가 호출."""
        self.history = []

# ──────────────────────────────────────────
# 2. GPT 플레이어 슬롯  ← GPT가 이 부분을 채웁니다
# ──────────────────────────────────────────
# --- GPT: 아래 클래스를 완성해 주세요 ---
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
# --- GPT 슬롯 끝 ---


# ──────────────────────────────────────────
# 3. Gemini 플레이어 슬롯  ← Gemini가 이 부분을 채웁니다
# ──────────────────────────────────────────
# --- Gemini: 아래 클래스를 완성해 주세요 ---
import random

class GeminiPlayer(BasePlayer):
    """
    4세대 알고리즘: Hedge 기반 연속 혼합 전략 (Hedge Algorithm with Longest Prefix Matching)
    
    최대 20깊이까지의 '가장 긴 일치 패턴'을 탐색하되, 단일 최고점 전략을 맹신하지 않습니다.
    각 메타 엔진의 승/무/패 성과에 따라 가중치를 실시간으로 부여하고, 그 가중치에 비례해
    확률적으로 패를 섞어냅니다. (혼합 전략 내시 균형 응용)
    """
    def __init__(self):
        super().__init__(name="Gemini")
        self.my_history = []
        self.op_history = []
        
        # 13개의 독립 메타 전략 (상대패턴3, 내패턴3, 복합패턴3, 빈도3, 무작위1)
        self.num_strategies = 13
        self.weights = [1.0] * self.num_strategies
        self.last_predictions = [random.randint(0, 2) for _ in range(self.num_strategies)]
        
        # 가중치 업데이트용 하이퍼파라미터
        self.win_bonus = 1.15      # 승리 시 가중치 증가
        self.draw_penalty = 0.90   # 무승부 시 약간의 페널티
        self.lose_penalty = 0.50   # 패배 시 대폭 페널티
        self.decay = 0.96          # 매 라운드 기본 노후화 (최신 메타 반영)

    def get_counter(self, move):
        """승리하는 카운터 패 반환 (ROCK:0 -> PAPER:2, SCISSORS:1 -> ROCK:0, PAPER:2 -> SCISSORS:1)"""
        return (move - 1) % 3

    def find_longest_match(self, seq, max_depth=20):
        """주어진 기록에서 가장 긴 과거 일치 패턴의 다음 패를 탐색"""
        if len(seq) < 2:
            return None
        for d in range(min(max_depth, len(seq) - 1), 0, -1):
            target = seq[-d:]
            for i in range(len(seq) - d - 1, -1, -1):
                if seq[i:i+d] == target:
                    return seq[i+d]
        return None

    def find_longest_comb_match(self, comb_seq, max_depth=20):
        """양측 복합 기록에서 가장 긴 일치 패턴 탐색"""
        if len(comb_seq) < 2:
            return None
        for d in range(min(max_depth, len(comb_seq) - 1), 0, -1):
            target = comb_seq[-d:]
            for i in range(len(comb_seq) - d - 1, -1, -1):
                if comb_seq[i:i+d] == target:
                    return self.op_history[i+d]
        return None

    def choose(self, history: list) -> int:
        if not self.my_history:
            move = random.randint(0, 2)
            self.last_predictions = [move] * self.num_strategies
            return move

        preds = []
        
        # 1. 상대방 기록 기반 (상대의 다음 패 예측 후 카운터, 역카운터 등 전개)
        p_op = self.find_longest_match(self.op_history)
        if p_op is None: p_op = random.randint(0, 2)
        preds.extend([self.get_counter(p_op), p_op, (p_op + 1) % 3]) 
        
        # 2. 내 기록 기반 (상대가 나의 다음 패를 예측한다고 가정)
        p_my = self.find_longest_match(self.my_history)
        if p_my is None: p_my = random.randint(0, 2)
        opp_guess = self.get_counter(p_my) # 상대가 낼 것이라 예상되는 패
        preds.extend([self.get_counter(opp_guess), opp_guess, (opp_guess + 1) % 3])
        
        # 3. 양측 복합 기록 기반 (고차원 콤보 패턴)
        comb = list(zip(self.my_history, self.op_history))
        p_comb = self.find_longest_comb_match(comb)
        if p_comb is None: p_comb = random.randint(0, 2)
        preds.extend([self.get_counter(p_comb), p_comb, (p_comb + 1) % 3])
        
        # 4. 단순 빈도 기반 (패턴이 붕괴되었을 때의 안전망)
        counts = [self.op_history.count(0), self.op_history.count(1), self.op_history.count(2)]
        p_freq = counts.index(max(counts)) if sum(counts) > 0 else random.randint(0, 2)
        preds.extend([self.get_counter(p_freq), p_freq, (p_freq + 1) % 3])
        
        # 5. 완전 무작위 전략 (최후의 보루)
        preds.append(random.randint(0, 2))

        self.last_predictions = preds

        # --- 혼합 전략 (룰렛 휠 선택) ---
        total_weight = sum(self.weights)
        if total_weight <= 0:
            return random.randint(0, 2)
            
        r = random.uniform(0, total_weight)
        cum = 0.0
        for i, w in enumerate(self.weights):
            cum += w
            if r <= cum:
                return self.last_predictions[i]
                
        return self.last_predictions[-1]

    def update(self, my_move: int, opp_move: int) -> None:
        for i, p in enumerate(self.last_predictions):
            # 1. 기본 노후화 적용
            self.weights[i] *= self.decay
            
            # 2. 결과에 따른 가중치 조정
            if p == self.get_counter(opp_move):   # 이기는 패를 냈을 경우
                self.weights[i] *= self.win_bonus
            elif p == opp_move:                   # 비기는 패를 냈을 경우
                self.weights[i] *= self.draw_penalty
            else:                                 # 지는 패를 냈을 경우
                self.weights[i] *= self.lose_penalty
                
            # 가중치가 0에 수렴하여 엔진이 죽는 것을 방지
            self.weights[i] = max(0.01, self.weights[i])

        self.my_history.append(my_move)
        self.op_history.append(opp_move)
# --- Gemini 슬롯 끝 ---


# ──────────────────────────────────────────
# 4. 대결 엔진 (수정 금지)
# ──────────────────────────────────────────
def run_match(player_a: BasePlayer, player_b: BasePlayer, n_rounds: int):
    """
    두 플레이어를 n_rounds 동안 대결시키고 라운드별 누적 승률을 반환.
    반환: (win_rate_a, win_rate_b, draw_rate) — 각각 길이 n_rounds의 리스트
    """
    player_a.reset()
    player_b.reset()

    wins_a = wins_b = draws = 0
    wr_a, wr_b, wr_d = [], [], []

    for r in range(1, n_rounds + 1):
        move_a = player_a.choose(list(player_a.history))
        move_b = player_b.choose(list(player_b.history))

        # 유효성 검사
        assert move_a in (ROCK, SCISSORS, PAPER), f"{player_a.name}: 잘못된 패 {move_a}"
        assert move_b in (ROCK, SCISSORS, PAPER), f"{player_b.name}: 잘못된 패 {move_b}"

        # 결과 판정
        if beats(move_a, move_b):
            wins_a += 1
        elif beats(move_b, move_a):
            wins_b += 1
        else:
            draws += 1

        # 각 플레이어에게 결과 통보 (상대 패 공개)
        player_a.update(move_a, move_b)
        player_b.update(move_b, move_a)

        # history 기록 (프레임워크가 관리)
        player_a.history.append((move_a, move_b))
        player_b.history.append((move_b, move_a))

        wr_a.append(wins_a / r)
        wr_b.append(wins_b / r)
        wr_d.append(draws / r)

    return wr_a, wr_b, wr_d


# ──────────────────────────────────────────
# 5. 시각화 (수정 금지)
# ──────────────────────────────────────────
def plot_results(wr_a, wr_b, wr_d, n_rounds, player_a_name, player_b_name):
    rounds = list(range(1, n_rounds + 1))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Rock Paper Scissors: {player_a_name} vs {player_b_name}\n({n_rounds} Rounds)", fontsize=14)

    # 상단: 누적 승률 추이
    ax = axes[0]
    ax.plot(rounds, [w * 100 for w in wr_a], label=player_a_name, color="#10a37f", linewidth=1.5)
    ax.plot(rounds, [w * 100 for w in wr_b], label=player_b_name, color="#4285f4", linewidth=1.5)
    ax.plot(rounds, [w * 100 for w in wr_d], label="Draw", color="#aaaaaa", linewidth=1, linestyle="--")
    ax.axhline(33.33, color="black", linestyle=":", linewidth=0.8, alpha=0.5, label="Random baseline (33.3%)")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 70)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 하단: 최종 결과 막대
    ax2 = axes[1]
    final = [wr_a[-1] * 100, wr_b[-1] * 100, wr_d[-1] * 100]
    colors = ["#10a37f", "#4285f4", "#aaaaaa"]
    bars = ax2.barh([player_a_name, player_b_name, "Draw"], final, color=colors)
    for bar, val in zip(bars, final):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=10)
    ax2.set_xlim(0, 80)
    ax2.set_xlabel("Final Win Rate (%)")
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("rps_result_v4.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n결과 이미지 저장됨: rps_result.png")


# ──────────────────────────────────────────
# 6. 다회 평균 실행 (통계적 신뢰성 확보)
# ──────────────────────────────────────────
def run_multi_trial(n_rounds: int, n_trials: int):
    """
    n_trials번 독립 실행 후 라운드별 승률을 평균냅니다.
    각 시행은 서로 다른 시드를 사용하며, 재현 가능하도록 시드를 고정합니다.
    """
    sum_a = [0.0] * n_rounds
    sum_b = [0.0] * n_rounds
    sum_d = [0.0] * n_rounds

    final_a_list, final_b_list, final_d_list = [], [], []

    for trial in range(n_trials):
        seed = 42 + trial  # 재현 가능한 시드 (42, 43, 44, ...)
        random.seed(seed)

        gpt    = GPTPlayer()
        gemini = GeminiPlayer()

        wr_a, wr_b, wr_d = run_match(gpt, gemini, n_rounds)

        for r in range(n_rounds):
            sum_a[r] += wr_a[r]
            sum_b[r] += wr_b[r]
            sum_d[r] += wr_d[r]

        final_a_list.append(wr_a[-1])
        final_b_list.append(wr_b[-1])
        final_d_list.append(wr_d[-1])

        print(f"  Trial {trial+1:>3d}/{n_trials}  |  "
              f"GPT {wr_a[-1]*100:.1f}%  Gemini {wr_b[-1]*100:.1f}%  Draw {wr_d[-1]*100:.1f}%")

    avg_a = [x / n_trials for x in sum_a]
    avg_b = [x / n_trials for x in sum_b]
    avg_d = [x / n_trials for x in sum_d]

    return avg_a, avg_b, avg_d, final_a_list, final_b_list, final_d_list


def plot_multi_results(avg_a, avg_b, avg_d, final_a, final_b, final_d,
                       n_rounds, n_trials, player_a_name, player_b_name):
    import statistics

    rounds = list(range(1, n_rounds + 1))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"Rock Paper Scissors: {player_a_name} vs {player_b_name}\n"
        f"({n_rounds} Rounds × {n_trials} Trials — Averaged)",
        fontsize=14
    )

    ax = axes[0]
    ax.plot(rounds, [w * 100 for w in avg_a], label=player_a_name, color="#10a37f", linewidth=1.8)
    ax.plot(rounds, [w * 100 for w in avg_b], label=player_b_name, color="#4285f4", linewidth=1.8)
    ax.plot(rounds, [w * 100 for w in avg_d], label="Draw", color="#aaaaaa", linewidth=1.2, linestyle="--")
    ax.axhline(33.33, color="black", linestyle=":", linewidth=0.8, alpha=0.5, label="Random baseline (33.3%)")
    ax.set_ylabel("Win Rate (%) — Trial Average")
    ax.set_ylim(0, 70)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    mean_a = statistics.mean(final_a) * 100
    mean_b = statistics.mean(final_b) * 100
    mean_d = statistics.mean(final_d) * 100
    std_a  = statistics.stdev(final_a) * 100 if n_trials > 1 else 0
    std_b  = statistics.stdev(final_b) * 100 if n_trials > 1 else 0
    std_d  = statistics.stdev(final_d) * 100 if n_trials > 1 else 0

    labels = [player_a_name, player_b_name, "Draw"]
    means  = [mean_a, mean_b, mean_d]
    stds   = [std_a,  std_b,  std_d]
    colors = ["#10a37f", "#4285f4", "#aaaaaa"]

    bars = ax2.barh(labels, means, xerr=stds, color=colors,
                    error_kw=dict(ecolor="black", capsize=4, linewidth=1.2))
    for bar, val, std in zip(bars, means, stds):
        ax2.text(bar.get_width() + std + 0.5,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}% ± {std:.1f}%", va="center", fontsize=9)

    ax2.set_xlim(0, 80)
    ax2.set_xlabel("Final Win Rate (%) — Mean ± Std")
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("rps_result_multi_v4.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n결과 이미지 저장됨: rps_result_multi.png")


if __name__ == "__main__":
    N_ROUNDS = 1000   # 라운드 수
    N_TRIALS = 50     # 시행 횟수 (많을수록 신뢰도 상승, 시간 증가)

    gpt_name    = GPTPlayer().name
    gemini_name = GeminiPlayer().name

    print(f"[심판: Claude] {gpt_name} vs {gemini_name}")
    print(f"  {N_ROUNDS} rounds × {N_TRIALS} trials\n")

    avg_a, avg_b, avg_d, fa, fb, fd = run_multi_trial(N_ROUNDS, N_TRIALS)

    import statistics
    print(f"\n=== 최종 평균 결과 ===")
    print(f"  {gpt_name:10s} 승률: {statistics.mean(fa)*100:.2f}% ± {statistics.stdev(fa)*100:.2f}%")
    print(f"  {gemini_name:10s} 승률: {statistics.mean(fb)*100:.2f}% ± {statistics.stdev(fb)*100:.2f}%")
    print(f"  무승부      : {statistics.mean(fd)*100:.2f}% ± {statistics.stdev(fd)*100:.2f}%")

    gpt_wins = sum(1 for a, b in zip(fa, fb) if a > b)
    print(f"\n  {N_TRIALS}번 중 {gpt_name} 승: {gpt_wins}회 / {gemini_name} 승: {N_TRIALS - gpt_wins}회")

    winner = gpt_name if statistics.mean(fa) > statistics.mean(fb) else gemini_name
    print(f"\n🏆 통계적 승자: {winner}")

    plot_multi_results(avg_a, avg_b, avg_d, fa, fb, fd,
                       N_ROUNDS, N_TRIALS, gpt_name, gemini_name)