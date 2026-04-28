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
# --- GPT 슬롯 끝 ---


# ──────────────────────────────────────────
# 3. Gemini 플레이어 슬롯  ← Gemini가 이 부분을 채웁니다
# ──────────────────────────────────────────
# --- Gemini: 아래 클래스를 완성해 주세요 ---
import random
import numpy as np

class GeminiPlayer(BasePlayer):
    """
    5세대 블라인드 최종 결전 알고리즘: 확률적 신경망 앙상블 (Probabilistic Neural Ensemble)
    
    Numpy를 활용하여 58개의 다중 깊이/메타 전략의 예측을 확률 분포 벡터로 처리합니다.
    매 라운드 상대방의 실제 패에 대한 교차 엔트로피(Cross-Entropy)를 계산하고,
    경사하강법을 통해 각 엔진의 가중치(Logits)를 실시간 최적화합니다.
    디리클레 스무딩(Dirichlet Smoothing)을 적용해 노이즈에 대한 극강의 내성을 가집니다.
    """
    def __init__(self):
        super().__init__(name="Gemini")
        self.my_history = []
        self.op_history = []
        
        self.max_depth = 6
        # 엔진 구성: (상대기록 6 + 내기록 6 + 복합기록 6) * 3(메타시프트) + 빈도 3 + 무작위 1 = 총 58개
        self.num_engines = 3 * self.max_depth * 3 + 3 + 1
        
        # 신경망의 Logit 가중치 초기화
        self.weights = np.zeros(self.num_engines)
        self.lr = 0.5        # 학습률 (Learning Rate)
        self.decay = 0.98    # 과거 정보 망각률 (상대의 급격한 전략 변화에 적응)
        
        # 직전 라운드에서 각 엔진이 예측한 [바위, 가위, 보] 확률 분포
        self.last_probs = np.ones((self.num_engines, 3)) / 3.0

    def get_dist(self, seq, depth):
        """특정 깊이의 패턴을 찾아 디리클레 스무딩이 적용된 확률 분포 반환"""
        dist = np.ones(3) * 0.1  # Smoothing prior
        if len(seq) <= depth:
            return dist
        
        target = tuple(seq[-depth:])
        for i in range(len(seq) - depth - 1, -1, -1):
            if tuple(seq[i:i+depth]) == target:
                dist[self.op_history[i+depth]] += 1.0
        return dist

    def get_comb_dist(self, comb, depth):
        """양측 복합 기록에서 패턴을 찾는 확률 분포"""
        dist = np.ones(3) * 0.1
        if len(comb) <= depth:
            return dist
            
        target = tuple(comb[-depth:])
        for i in range(len(comb) - depth - 1, -1, -1):
            if tuple(comb[i:i+depth]) == target:
                dist[self.op_history[i+depth]] += 1.0
        return dist

    def choose(self, history: list) -> int:
        if not self.my_history:
            return random.randint(0, 2)

        all_probs = []
        
        # 1. 상대방 단독 패턴 기반 확률
        for d in range(1, self.max_depth + 1):
            dist = self.get_dist(self.op_history, d)
            p = dist / np.sum(dist)
            all_probs.extend([np.roll(p, 0), np.roll(p, 1), np.roll(p, 2)])
            
        # 2. 나의 단독 패턴 기반 확률 (상대가 내 패턴을 읽는다고 가정)
        for d in range(1, self.max_depth + 1):
            dist = self.get_dist(self.my_history, d)
            p = dist / np.sum(dist)
            all_probs.extend([np.roll(p, 0), np.roll(p, 1), np.roll(p, 2)])
            
        # 3. 양측 콤보 패턴 기반 확률
        comb = list(zip(self.my_history, self.op_history))
        for d in range(1, self.max_depth + 1):
            dist = self.get_comb_dist(comb, d)
            p = dist / np.sum(dist)
            all_probs.extend([np.roll(p, 0), np.roll(p, 1), np.roll(p, 2)])
            
        # 4. 단순 빈도 기반 (최후의 확률적 안전망)
        counts = np.array([self.op_history.count(0), self.op_history.count(1), self.op_history.count(2)]) + 0.1
        p_freq = counts / np.sum(counts)
        all_probs.extend([np.roll(p_freq, 0), np.roll(p_freq, 1), np.roll(p_freq, 2)])
        
        # 5. 완전 무작위 엔진
        all_probs.append(np.array([1/3, 1/3, 1/3]))
        
        self.last_probs = np.array(all_probs)
        
        # Softmax를 통해 Logit 가중치를 엔진별 신뢰도(확률)로 변환
        exp_w = np.exp(self.weights - np.max(self.weights))
        meta_probs = exp_w / np.sum(exp_w)
        
        # 58개 엔진의 예측을 신뢰도에 따라 가중 평균 (최종 상대 패 확률 분포)
        final_opp_probs = np.sum(self.last_probs * meta_probs[:, np.newaxis], axis=0)
        
        # 가장 확률이 높은 상대의 패를 예측
        predicted_opp = np.argmax(final_opp_probs)
        
        # ROCK(0)->PAPER(2), SCISSORS(1)->ROCK(0), PAPER(2)->SCISSORS(1)
        return int((predicted_opp - 1) % 3)

    def update(self, my_move: int, opp_move: int) -> None:
        if self.my_history:
            # 방금 상대가 낸 패(opp_move)에 대해, 각 엔진이 사전에 부여했던 예측 확률값
            actual_probs = self.last_probs[:, opp_move]
            actual_probs = np.clip(actual_probs, 1e-7, 1.0) # log(0) 방지
            
            # Gradient Ascent: 실제 결과를 잘 예측한 엔진의 Logit을 높임
            self.weights = self.weights * self.decay + self.lr * np.log(actual_probs)

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
    plt.savefig("rps_result_v5.png", dpi=150, bbox_inches="tight")
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
    plt.savefig("rps_result_multi_v5.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n결과 이미지 저장됨: rps_result_multi.png")


if __name__ == "__main__":
    N_ROUNDS = 5000   # 라운드 수
    N_TRIALS = 10     # 시행 횟수 (많을수록 신뢰도 상승, 시간 증가)

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