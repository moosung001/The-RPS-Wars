# 내용을 여기에 채워주세요
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