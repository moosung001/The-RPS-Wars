# 내용을 여기에 채워주세요
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