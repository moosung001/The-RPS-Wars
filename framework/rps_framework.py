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

        self.opp_counts = [0, 0, 0]
        self.markov1 = {}
        self.markov2 = {}
        self.by_pair = {}
        self.by_my_last = {}

        self.model_scores = {}
        self.last_model_probs = {}

    def _one_hot_probs(self, move, sharp=0.86):
        off = (1.0 - sharp) / 2.0
        p = [off, off, off]
        p[move] = sharp
        return p

    def _normalize(self, xs):
        s = float(sum(xs))
        if s <= 0:
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        return [x / s for x in xs]

    def _smoothed_probs(self, counts, alpha=1.0):
        total = counts[0] + counts[1] + counts[2] + 3.0 * alpha
        return [(counts[i] + alpha) / total for i in range(3)]

    def _blend(self, p, q, w=0.75):
        return [w * p[i] + (1.0 - w) * q[i] for i in range(3)]

    def _get_table_probs(self, table, key, backoff, alpha=1.0):
        counts = table.get(key)
        if counts is None:
            return list(backoff)
        return self._blend(self._smoothed_probs(counts, alpha), backoff, 0.75)

    def _recent_probs(self, horizon=20, gamma=0.85):
        if not self.opp_hist:
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

        counts = [0.0, 0.0, 0.0]
        weight = 1.0
        for move in reversed(self.opp_hist[-horizon:]):
            counts[move] += weight
            weight *= gamma

        total = sum(counts) + 3.0
        return [(counts[i] + 1.0) / total for i in range(3)]

    def _record(self, table, key, nxt_move):
        if key not in table:
            table[key] = [0, 0, 0]
        table[key][nxt_move] += 1

    def choose(self, history) -> int:
        import math

        n = len(self.opp_hist)

        global_p = self._smoothed_probs(self.opp_counts, alpha=1.0)
        recent_p = self._recent_probs()
        base_p = self._blend(recent_p, global_p, 0.65)

        models = {
            "global": global_p,
            "recent": recent_p,
        }

        if n >= 1:
            opp_last = self.opp_hist[-1]
            my_last = self.my_hist[-1]

            models["markov1_opp"] = self._get_table_probs(self.markov1, opp_last, base_p)
            models["by_my_last"] = self._get_table_probs(self.by_my_last, my_last, base_p)
            models["pair_last"] = self._get_table_probs(self.by_pair, (my_last, opp_last), base_p)

            models["repeat_opp"] = self._one_hot_probs(opp_last, 0.86)
            models["cycle_up"] = self._one_hot_probs((opp_last + 1) % 3, 0.86)
            models["cycle_down"] = self._one_hot_probs((opp_last + 2) % 3, 0.86)
            models["copy_me"] = self._one_hot_probs(my_last, 0.86)
            models["beat_my_last"] = self._one_hot_probs((my_last + 2) % 3, 0.86)
            models["lose_to_my_last"] = self._one_hot_probs((my_last + 1) % 3, 0.86)

        if n >= 2:
            models["markov2_opp"] = self._get_table_probs(
                self.markov2,
                (self.opp_hist[-2], self.opp_hist[-1]),
                base_p
            )

        mixed = [0.0, 0.0, 0.0]
        for name, probs in models.items():
            score = self.model_scores.get(name, 0.0)
            weight = math.exp(max(-4.0, min(4.0, score)) * 1.6)
            for i in range(3):
                mixed[i] += weight * probs[i]

        p_opp = self._normalize(mixed)

        if n < 3:
            eps = 0.22
        elif n < 8:
            eps = 0.12
        elif n < 20:
            eps = 0.05
        else:
            eps = 0.02

        p_opp = [(1.0 - eps) * p_opp[i] + eps / 3.0 for i in range(3)]
        self.last_model_probs = models

        payoffs = [
            p_opp[1] - p_opp[2],
            p_opp[2] - p_opp[0],
            p_opp[0] - p_opp[1],
        ]

        if max(p_opp) - min(p_opp) < 0.04:
            return random.choice([0, 1, 2])

        best_value = max(payoffs)
        candidates = [i for i, v in enumerate(payoffs) if abs(v - best_value) < 1e-12]
        return random.choice(candidates)

    def update(self, my_move: int, opp_move: int) -> None:
        for name, probs in self.last_model_probs.items():
            reward = probs[opp_move] - (1.0 / 3.0)
            old = self.model_scores.get(name, 0.0)
            self.model_scores[name] = 0.90 * old + 1.35 * reward

        n = len(self.opp_hist)

        if n >= 1:
            self._record(self.markov1, self.opp_hist[-1], opp_move)
            self._record(self.by_my_last, self.my_hist[-1], opp_move)
            self._record(self.by_pair, (self.my_hist[-1], self.opp_hist[-1]), opp_move)

        if n >= 2:
            self._record(self.markov2, (self.opp_hist[-2], self.opp_hist[-1]), opp_move)

        self.opp_counts[opp_move] += 1
        self.my_hist.append(my_move)
        self.opp_hist.append(opp_move)
# --- GPT 슬롯 끝 ---


# ──────────────────────────────────────────
# 3. Gemini 플레이어 슬롯  ← Gemini가 이 부분을 채웁니다
# ──────────────────────────────────────────
# --- Gemini: 아래 클래스를 완성해 주세요 ---
class GeminiPlayer(BasePlayer):
    """
    다중 깊이 마르코프 체인(N-gram) 기반의 패턴 인식 알고리즘을 사용합니다.
    최대 6단계까지의 과거 기록(나의 패, 상대의 패)을 컨텍스트로 삼아 
    상대방의 다음 패를 예측하고, 이를 카운터하는 패를 냅니다.
    """
    def __init__(self):
        super().__init__(name="Gemini")
        self.max_depth = 6
        # 패턴 메모리: context 튜플 -> 상대방의 [바위, 가위, 보] 등장 횟수 리스트
        self.ngrams = {}
        # 프레임워크의 history와 별개로 (my_move, opp_move)를 추적하기 위한 내부 리스트
        self.internal_history = []

    def choose(self, history: list) -> int:
        ROCK, SCISSORS, PAPER = 0, 1, 2
        
        # 첫 라운드거나 기록이 없으면 랜덤
        if not self.internal_history:
            return random.choice([ROCK, SCISSORS, PAPER])

        # 가장 긴 컨텍스트(최대 깊이)부터 짧은 컨텍스트 순으로 하향 탐색
        for depth in range(min(self.max_depth, len(self.internal_history)), 0, -1):
            context = tuple(self.internal_history[-depth:])
            if context in self.ngrams:
                counts = self.ngrams[context]
                max_count = max(counts)
                
                # 예측된 상대의 다음 패 (동률일 경우 패턴 고착화를 막기 위해 랜덤 선택)
                likely_opp_moves = [i for i, c in enumerate(counts) if c == max_count]
                predicted_opp_move = random.choice(likely_opp_moves)
                
                # 예측한 상대 패를 이기는 패를 반환
                # ROCK(0)은 PAPER(2)가, SCISSORS(1)는 ROCK(0)이, PAPER(2)는 SCISSORS(1)가 이김
                return (predicted_opp_move - 1) % 3

        # 일치하는 패턴이 없는 경우: 전체 기록 기반 빈도 분석 (가장 많이 낸 패를 카운터)
        if self.internal_history:
            opp_moves = [move[1] for move in self.internal_history]
            counts = [opp_moves.count(ROCK), opp_moves.count(SCISSORS), opp_moves.count(PAPER)]
            predicted_opp_move = counts.index(max(counts))
            return (predicted_opp_move - 1) % 3

        return random.choice([ROCK, SCISSORS, PAPER])

    def update(self, my_move: int, opp_move: int) -> None:
        # 방금 나온 상대의 패(opp_move)를 이전 컨텍스트들의 결과로 학습
        if self.internal_history:
            for depth in range(1, min(self.max_depth, len(self.internal_history)) + 1):
                context = tuple(self.internal_history[-depth:])
                if context not in self.ngrams:
                    self.ngrams[context] = [0, 0, 0]
                self.ngrams[context][opp_move] += 1
                
        # 현재 라운드 결과를 내부 기록에 추가
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
    plt.savefig("rps_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n결과 이미지 저장됨: rps_result.png")


# ──────────────────────────────────────────
# 6. 메인 실행
# ──────────────────────────────────────────
if __name__ == "__main__":
    N_ROUNDS = 1000  # 원하는 라운드 수로 변경 가능

    gpt    = GPTPlayer()
    gemini = GeminiPlayer()

    print(f"[심판: Claude] {gpt.name} vs {gemini.name} — {N_ROUNDS}라운드 시작!")
    wr_a, wr_b, wr_d = run_match(gpt, gemini, N_ROUNDS)

    print(f"\n=== 최종 결과 ===")
    print(f"  {gpt.name:10s} 승률: {wr_a[-1]*100:.1f}%")
    print(f"  {gemini.name:10s} 승률: {wr_b[-1]*100:.1f}%")
    print(f"  무승부      : {wr_d[-1]*100:.1f}%")
    winner = gpt.name if wr_a[-1] > wr_b[-1] else (gemini.name if wr_b[-1] > wr_a[-1] else "무승부")
    print(f"\n🏆 승자: {winner}")

    plot_results(wr_a, wr_b, wr_d, N_ROUNDS, gpt.name, gemini.name)