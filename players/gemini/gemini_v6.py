import random
import numpy as np

class GeminiPlayer(BasePlayer):
    """
    6세대 알고리즘: 다중 지수 가중치 (Multi-Hedge) & 텐서 앙상블
    
    GPT의 방대한 휴리스틱 코드는 특정 조건에서 고정된 ε-greedy 임계값을 가지는 취약점이 있습니다.
    이 알고리즘은 GPT의 패턴 수렴 궤적을 메타-학습하는 '동적 학습률 Hedge' 모델과,
    알려지지 않은 Claude의 공격을 방어할 최적화된 혼합 내시(Mixed Nash) 확률망을 결합했습니다.
    """
    def __init__(self):
        super().__init__(name="Gemini")
        self.reset()

    def reset(self):
        super().reset()
        self.my_hist = []
        self.opp_hist = []
        
        self.max_depth = 7
        self.meta_shifts = 3
        # 예측 엔진 생성: (상대 기록 7 + 내 기록 7 + 복합 기록 7) * 3(메타시프트) + 빈도 3 + 완전 무작위 1
        self.num_engines = 3 * self.max_depth * self.meta_shifts + 4
        
        # 수치 안정성을 위해 Log 가중치 배열 사용
        self.log_weights = np.zeros(self.num_engines)
        self.last_predictions = np.zeros(self.num_engines, dtype=int)
        
        self.eta = 2.0      # 학습률 (최근 트렌드에 극도로 민감하게 반응)
        self.decay = 0.95   # 노후화 (GPT의 4단계 decay rate를 포괄하여 무력화)

    def _find_pattern(self, seq, depth):
        """단일 기록에서 패턴 탐색"""
        if len(seq) <= depth:
            return None
        target = tuple(seq[-depth:])
        # 뒤에서부터 스캔하여 가장 최근 일치 항목 반환
        for i in range(len(seq) - depth - 1, -1, -1):
            if tuple(seq[i:i+depth]) == target:
                return seq[i+depth]
        return None

    def _find_pair_pattern(self, my_seq, opp_seq, depth):
        """나와 상대의 교차 기록에서 패턴 탐색"""
        if len(my_seq) <= depth:
            return None
        target_m = tuple(my_seq[-depth:])
        target_o = tuple(opp_seq[-depth:])
        for i in range(len(my_seq) - depth - 1, -1, -1):
            if tuple(my_seq[i:i+depth]) == target_m and tuple(opp_seq[i:i+depth]) == target_o:
                return opp_seq[i+depth] # 일치 시 상대의 다음 패 반환
        return None

    def choose(self, history: list) -> int:
        ROCK, SCISSORS, PAPER = 0, 1, 2
        
        if len(self.opp_hist) < 2:
            move = random.randint(0, 2)
            self.last_predictions.fill(move)
            return move

        preds = []
        
        # 1. 상대방 단일 패턴 탐색
        for d in range(1, self.max_depth + 1):
            p = self._find_pattern(self.opp_hist, d)
            if p is None: p = random.randint(0, 2)
            preds.extend([(p+2)%3, p, (p+1)%3]) # 승리(직접 카운터), 비김, 패배(역이용)
            
        # 2. 나의 단일 패턴 탐색 (상대가 나를 읽을 때의 역카운터)
        for d in range(1, self.max_depth + 1):
            p = self._find_pattern(self.my_hist, d)
            if p is None: p = random.randint(0, 2)
            opp_guess = (p+2)%3 
            preds.extend([(opp_guess+2)%3, opp_guess, (opp_guess+1)%3])
            
        # 3. 쌍(Pair) 복합 패턴 탐색
        for d in range(1, self.max_depth + 1):
            p = self._find_pair_pattern(self.my_hist, self.opp_hist, d)
            if p is None: p = random.randint(0, 2)
            preds.extend([(p+2)%3, p, (p+1)%3])
            
        # 4. 전체 빈도 기반 및 안전망 (Fallback)
        counts = [self.opp_hist.count(ROCK), self.opp_hist.count(SCISSORS), self.opp_hist.count(PAPER)]
        freq_p = np.argmax(counts) if sum(counts) > 0 else random.randint(0, 2)
        preds.extend([(freq_p+2)%3, freq_p, (freq_p+1)%3])
        
        # 5. 완전 난수 엔진
        preds.append(random.randint(0, 2))
        
        self.last_predictions = np.array(preds)
        
        # Softmax 연산으로 Log 가중치를 채택 확률(Probabilities)로 변환
        max_lw = np.max(self.log_weights)
        exp_w = np.exp(self.log_weights - max_lw)
        probs = exp_w / np.sum(exp_w)
        
        # 계산된 확률에 기반해 룰렛 휠 선택 실행 (완벽한 혼합 내시 균형 추구)
        chosen_engine_idx = np.random.choice(self.num_engines, p=probs)
        return int(self.last_predictions[chosen_engine_idx])

    def update(self, my_move: int, opp_move: int) -> None:
        if self.my_hist:
            # 풀 정보(Full-information) 게임에 맞춘 보상 벡터화
            rewards = np.zeros(self.num_engines)
            for i, p in enumerate(self.last_predictions):
                if p == (opp_move + 2) % 3:
                    rewards[i] = 1.0    # 이기는 예측
                elif p == opp_move:
                    rewards[i] = 0.0    # 비기는 예측
                else:
                    rewards[i] = -1.0   # 지는 예측
            
            # Hedge 업데이트 규칙: 기존 가중치 노후화 후 새로운 보상 적용
            self.log_weights = self.log_weights * self.decay + self.eta * rewards
            
        # 프레임워크 히스토리와 별개로 빠른 탐색을 위해 내부 캐싱
        self.my_hist.append(my_move)
        self.opp_hist.append(opp_move)