# 내용을 여기에 채워주세요
import random

class GeminiPlayer(BasePlayer):
    """
    3세대 알고리즘: 다중 무장 상금 기계(MAB) 기반 Iocaine 아키텍처
    
    상대/나/양측의 복합 패턴을 깊이 1~5까지 추적하는 15개의 예측기를 가동합니다.
    각 예측기마다 3가지 메타 시프트(직접 카운터, 상대의 카운터를 역이용, 역이용의 역이용)를
    모두 계산하여 총 45개의 패턴 전략과 3개의 빈도 전략, 1개의 랜덤 전략(총 49개)을 경쟁시킵니다.
    """
    def __init__(self):
        super().__init__(name="Gemini")
        self.my_history = []
        self.op_history = []
        
        self.num_strategies = 49
        self.scores = [0.0] * self.num_strategies
        self.last_predictions = [random.randint(0, 2) for _ in range(self.num_strategies)]
        self.decay = 0.90  # 최신 메타를 빠르게 따라가기 위한 감쇠율

    def predict_next(self, seq, d):
        """단일 기록에서 패턴 탐색"""
        if len(seq) <= d:
            return None
        target = seq[-d:]
        for i in range(len(seq) - d - 1, -1, -1):
            if seq[i:i+d] == target:
                return seq[i+d]
        return None
        
    def predict_next_comb(self, comb, d):
        """양측 복합 기록에서 상대의 다음 패 탐색"""
        if len(comb) <= d:
            return None
        target = comb[-d:]
        for i in range(len(comb) - d - 1, -1, -1):
            if comb[i:i+d] == target:
                return self.op_history[i+d]  # 복합 패턴 일치 시, 당시 상대가 냈던 패 반환
        return None

    def choose(self, history: list) -> int:
        if not self.my_history:
            move = random.randint(0, 2)
            self.last_predictions = [move] * self.num_strategies
            return move

        preds = []
        
        # 1. 상대 패턴 기반 엔진 (깊이 1~5 x 3 메타 시프트)
        for d in range(1, 6):
            p = self.predict_next(self.op_history, d)
            if p is None: p = random.randint(0, 2)
            preds.extend([(p+2)%3, p, (p+1)%3]) 
            
        # 2. 내 패턴 기반 엔진 (깊이 1~5 x 3 메타 시프트)
        # (내가 낼 패를 상대가 읽는다고 가정)
        for d in range(1, 6):
            p = self.predict_next(self.my_history, d)
            if p is None: p = random.randint(0, 2)
            preds.extend([(p+2)%3, p, (p+1)%3])
            
        # 3. 양측 복합 패턴 엔진 (깊이 1~5 x 3 메타 시프트)
        comb = list(zip(self.my_history, self.op_history))
        for d in range(1, 6):
            p = self.predict_next_comb(comb, d)
            if p is None: p = random.randint(0, 2)
            preds.extend([(p+2)%3, p, (p+1)%3])

        # 4. 전체 빈도 기반 엔진 (3 메타 시프트)
        counts = [self.op_history.count(0), self.op_history.count(1), self.op_history.count(2)]
        p = counts.index(max(counts))
        preds.extend([(p+2)%3, p, (p+1)%3])

        # 5. 완전 무작위 (안전망)
        preds.append(random.randint(0, 2))

        self.last_predictions = preds

        # 가장 신뢰도가 높은 전략 채택
        best_score = max(self.scores)
        if best_score <= 0:
            return random.randint(0, 2)  # 모든 엔진이 실패 중이면 무작위 회피
        
        best_idx = self.scores.index(best_score)
        return self.last_predictions[best_idx]

    def update(self, my_move: int, opp_move: int) -> None:
        # 지난 라운드에 예측했던 49개 전략을 실제 상대의 패를 기준으로 채점
        for i, p in enumerate(self.last_predictions):
            self.scores[i] *= self.decay
            
            if p == (opp_move + 2) % 3:   # 승리하는 패를 예측했는가?
                self.scores[i] += 1.2
            elif p == opp_move:           # 비기는 패를 예측했는가?
                self.scores[i] -= 0.5
            else:                         # 지는 패를 예측했는가?
                self.scores[i] -= 1.0
                
        # 내부 기록 업데이트
        self.my_history.append(my_move)
        self.op_history.append(opp_move)