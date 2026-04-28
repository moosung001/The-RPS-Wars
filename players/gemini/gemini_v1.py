# 내용을 여기에 채워주세요
import random

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