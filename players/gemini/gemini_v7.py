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