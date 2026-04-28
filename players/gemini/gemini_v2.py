# 내용을 여기에 채워주세요
import random

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