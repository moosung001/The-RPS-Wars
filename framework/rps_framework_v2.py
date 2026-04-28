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
        self.pair = {}
        self.out_opp = {}
        self.out_my = {}

        self.base_names = [
            "opp_last",
            "opp_freq",
            "opp_recent",
            "opp_m1",
            "opp_m2",
            "pair",
            "out_opp",
            "cycle_up",
            "cycle_down",
            "my_last",
            "my_freq",
            "my_recent",
            "my_m1",
            "my_m2",
            "out_my",
        ]

        self.experts = [(name, rot) for name in self.base_names for rot in (0, 1, 2)]
        self.logw = [0.0] * len(self.experts)
        self.last_actions = [0] * len(self.experts)

        self.recent_results = []
        self.last_probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    def _payoff(self, a, b):
        if (a == ROCK and b == SCISSORS) or (a == SCISSORS and b == PAPER) or (a == PAPER and b == ROCK):
            return 1
        if (b == ROCK and a == SCISSORS) or (b == SCISSORS and a == PAPER) or (b == PAPER and a == ROCK):
            return -1
        return 0

    def _decay_add(self, arr, move, gamma=0.85):
        for i in range(3):
            arr[i] *= gamma
        arr[move] += 1.0

    def _argmax_rand(self, xs):
        m = max(xs)
        cand = [i for i, x in enumerate(xs) if x == m]
        return random.choice(cand)

    def _table_pick(self, table, key, fallback):
        if key not in table:
            return fallback
        return self._argmax_rand(table[key])

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

        fallback_opp = self._argmax_rand(self.opp_counts)
        fallback_my = self._argmax_rand(self.my_counts) if sum(self.my_counts) > 0 else random.randrange(3)

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

        return random.randrange(3)

    def choose(self, history: list) -> int:
        import math

        n = len(self.opp_hist)
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

            w = math.exp(lw)
            masses[action] += w

        self.last_actions = acts

        s = sum(masses)
        if s <= 0:
            probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        else:
            probs = [m / s for m in masses]

        recent_window = self.recent_results[-30:]
        recent_avg = sum(recent_window) / len(recent_window) if recent_window else 0.0
        spread = max(probs) - min(probs)

        eps = 0.06 + max(0.0, -recent_avg) * 0.45

        if n < 15:
            eps += 0.18
        elif n < 60:
            eps += 0.10
        elif n < 150:
            eps += 0.05

        if spread < 0.12:
            eps += 0.08

        if eps < 0.08:
            eps = 0.08
        elif eps > 0.35:
            eps = 0.35

        probs = [(1.0 - eps) * p + eps / 3.0 for p in probs]
        self.last_probs = probs

        r = random.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c:
                return i

        return PAPER

    def update(self, my_move: int, opp_move: int) -> None:
        import math

        eta = 0.22
        decay = 0.97

        for i, action in enumerate(self.last_actions):
            reward = self._payoff(action, opp_move)
            self.logw[i] = decay * self.logw[i] + eta * reward

        n = len(self.opp_hist)

        if n >= 1:
            self.opp_m1.setdefault(self.opp_hist[-1], [0, 0, 0])[opp_move] += 1
            self.my_m1.setdefault(self.my_hist[-1], [0, 0, 0])[my_move] += 1
            self.pair.setdefault((self.my_hist[-1], self.opp_hist[-1]), [0, 0, 0])[opp_move] += 1

            prev_outcome = self._last_outcome()
            self.out_opp.setdefault(prev_outcome, [0, 0, 0])[opp_move] += 1
            self.out_my.setdefault(prev_outcome, [0, 0, 0])[my_move] += 1

        if n >= 2:
            self.opp_m2.setdefault((self.opp_hist[-2], self.opp_hist[-1]), [0, 0, 0])[opp_move] += 1
            self.my_m2.setdefault((self.my_hist[-2], self.my_hist[-1]), [0, 0, 0])[my_move] += 1

        self.opp_counts[opp_move] += 1
        self.my_counts[my_move] += 1

        self._decay_add(self.opp_recent, opp_move, gamma=0.85)
        self._decay_add(self.my_recent, my_move, gamma=0.85)

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
class GeminiPlayer(BasePlayer):
    """
    앙상블 메타 전략 (Meta-Strategy Ensemble) 모델
    
    여러 가지의 예측 엔진을 동시에 실행하고, 매 라운드 각 엔진의 가상 승률을 추적합니다.
    실시간으로 가장 성적이 좋은 엔진의 예측값을 채택하여 상대의 진화에 유연하게 대처합니다.
    """
    def __init__(self):
        super().__init__(name="Gemini")
        self.internal_history = []
        
        # 예측 엔진 정의
        # 0: 기본 N-gram (상대의 패턴을 읽음)
        # 1: 역 N-gram (상대가 내 패턴을 읽는다고 가정하고 그 카운터의 카운터를 침)
        # 2: 빈도 기반 분석 (패턴이 무너졌을 때의 안전망)
        self.strategies_count = 3
        self.strategy_scores = [0.0] * self.strategies_count
        self.last_predictions = [None] * self.strategies_count
        
        self.max_depth = 5
        self.opp_ngrams = {}
        self.my_ngrams = {}

    def _get_counter_move(self, predicted_move):
        """예측된 패를 이기는 카운터 패 반환"""
        # ROCK(0)->PAPER(2), SCISSORS(1)->ROCK(0), PAPER(2)->SCISSORS(1)
        return (predicted_move - 1) % 3

    def _predict_ngram(self, ngram_dict, sequence):
        """N-gram 사전을 바탕으로 다음 패 예측"""
        for depth in range(min(self.max_depth, len(sequence) - 1), 0, -1):
            context = tuple(sequence[-depth:])
            if context in ngram_dict:
                counts = ngram_dict[context]
                # 동률일 경우 고착화 방지를 위해 랜덤 픽
                max_count = max(counts)
                candidates = [i for i, c in enumerate(counts) if c == max_count]
                return random.choice(candidates)
        return None

    def choose(self, history: list) -> int:
        ROCK, SCISSORS, PAPER = 0, 1, 2
        
        if len(self.internal_history) < 2:
            move = random.choice([ROCK, SCISSORS, PAPER])
            self.last_predictions = [move] * self.strategies_count
            return move

        # --- 엔진 0: 상대방 패턴 예측 (Opponent N-gram) ---
        opp_seq = [h[1] for h in self.internal_history]
        pred_opp = self._predict_ngram(self.opp_ngrams, opp_seq)
        engine_0_move = self._get_counter_move(pred_opp) if pred_opp is not None else random.randint(0, 2)
        
        # --- 엔진 1: 나의 패턴 예측 후 역카운터 (My N-gram Meta-counter) ---
        # 1. 상대가 내가 다음에 낼 패(my_pred)를 예측한다고 가정
        # 2. 상대는 그것을 잡기 위해 opp_counter = 카운터(my_pred) 를 낼 것임
        # 3. 나는 그 상대의 패를 잡기 위해 카운터(opp_counter) 를 냄
        my_seq = [h[0] for h in self.internal_history]
        my_pred = self._predict_ngram(self.my_ngrams, my_seq)
        if my_pred is not None:
            opp_expected_counter = self._get_counter_move(my_pred)
            engine_1_move = self._get_counter_move(opp_expected_counter)
        else:
            engine_1_move = random.randint(0, 2)

        # --- 엔진 2: 단순 빈도 기반 분석 (Frequency Fallback) ---
        counts = [opp_seq.count(ROCK), opp_seq.count(SCISSORS), opp_seq.count(PAPER)]
        pred_freq = counts.index(max(counts))
        engine_2_move = self._get_counter_move(pred_freq)

        # 각 엔진의 제안 저장
        self.last_predictions = [engine_0_move, engine_1_move, engine_2_move]

        # 가장 점수가 높은 엔진 선택
        best_strategy_idx = self.strategy_scores.index(max(self.strategy_scores))
        
        # 모든 엔진의 점수가 0 이하라면 패턴이 꼬인 상태이므로 완전 무작위(Exploration) 시도
        if self.strategy_scores[best_strategy_idx] <= 0:
            return random.choice([ROCK, SCISSORS, PAPER])
            
        return self.last_predictions[best_strategy_idx]

    def update(self, my_move: int, opp_move: int) -> None:
        ROCK, SCISSORS, PAPER = 0, 1, 2
        
        # 1. 지난 라운드 엔진들의 성적 채점 및 노후화(Decay) 적용
        # 최신 트렌드를 더 중요하게 여기기 위해 점수를 매 라운드 감소시킴
        for i in range(self.strategies_count):
            self.strategy_scores[i] *= 0.9 
            
            predicted_move = self.last_predictions[i]
            if predicted_move is not None:
                if (predicted_move == ROCK and opp_move == SCISSORS) or \
                   (predicted_move == SCISSORS and opp_move == PAPER) or \
                   (predicted_move == PAPER and opp_move == ROCK):
                    self.strategy_scores[i] += 1.0  # 승리
                elif predicted_move != opp_move:
                    self.strategy_scores[i] -= 1.0  # 패배

        # 2. N-gram 패턴 메모리 업데이트
        if len(self.internal_history) > 0:
            opp_seq = [h[1] for h in self.internal_history] + [opp_move]
            my_seq = [h[0] for h in self.internal_history] + [my_move]
            
            for depth in range(1, min(self.max_depth, len(self.internal_history)) + 1):
                # 상대방 기록 학습
                opp_context = tuple(opp_seq[-depth-1:-1])
                if opp_context not in self.opp_ngrams:
                    self.opp_ngrams[opp_context] = [0, 0, 0]
                self.opp_ngrams[opp_context][opp_move] += 1
                
                # 나의 기록 학습
                my_context = tuple(my_seq[-depth-1:-1])
                if my_context not in self.my_ngrams:
                    self.my_ngrams[my_context] = [0, 0, 0]
                self.my_ngrams[my_context][my_move] += 1

        # 3. 내부 히스토리 저장
        self.internal_history.append((my_move, opp_move))
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
    plt.savefig("rps_result_v2.png", dpi=150, bbox_inches="tight")
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
    plt.savefig("rps_result_multi.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n결과 이미지 저장됨: rps_result_multi.png")


if __name__ == "__main__":
    N_ROUNDS = 1000   # 라운드 수
    N_TRIALS = 30     # 시행 횟수 (많을수록 신뢰도 상승, 시간 증가)

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