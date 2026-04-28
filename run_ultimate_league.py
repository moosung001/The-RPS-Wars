import os
import csv
import random
import math
import numpy as np
import itertools
import statistics
import time
from abc import ABC, abstractmethod

# ──────────────────────────────────────────
# 0. 공통 상수 및 베이스 클래스
# ──────────────────────────────────────────
ROCK, SCISSORS, PAPER = 0, 1, 2

def beats(a, b):
    return (a == ROCK and b == SCISSORS) or \
           (a == SCISSORS and b == PAPER) or \
           (a == PAPER and b == ROCK)

class BasePlayer(ABC):
    def __init__(self, name: str):
        self.name = name
        self.history = []
    @abstractmethod
    def choose(self, history: list) -> int:
        pass
    @abstractmethod
    def update(self, my_move: int, opp_move: int) -> None:
        pass
    def reset(self):
        self.history = []

# ──────────────────────────────────────────
# 1. 파일 동적 로더 (강화판)
# ──────────────────────────────────────────
def load_player_class(filepath, filename):
    """
    파일을 읽어들여 BasePlayer를 상속받은 클래스를 '동적'으로 찾아냅니다.
    클래스 이름이 바뀌어도 무조건 찾아내며, 파일이 비어있으면 스킵합니다.
    """
    namespace = {
        'BasePlayer': BasePlayer,
        'random': random,
        'np': np,
        'numpy': np,
        'math': math,
        'ROCK': ROCK, 'SCISSORS': SCISSORS, 'PAPER': PAPER
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read().strip()
        
    if not code:
        raise ValueError(f"[{filename}] 파일이 비어있습니다.")
        
    # 코드 실행하여 namespace에 클래스 등록
    code = code.replace("```python", "").replace("```", "")
    exec(code, namespace)
        
    # namespace를 뒤져서 BasePlayer를 상속받은 메인 클래스 추출
    for key, val in namespace.items():
        if isinstance(val, type) and issubclass(val, BasePlayer) and val is not BasePlayer:
            # 내부 클래스(Shadow 등)가 아닌 진짜 플레이어 클래스 반환
            if not key.startswith('_'): 
                return val
            
    raise ValueError(f"[{filename}] 안에 BasePlayer를 상속받은 클래스가 없습니다.")

# ──────────────────────────────────────────
# 2. 매치 및 리그 엔진
# ──────────────────────────────────────────
def run_match(player_a, player_b, n_rounds):
    player_a.reset()
    player_b.reset()
    wa, wb, draws = 0, 0, 0
    
    for _ in range(n_rounds):
        move_a = player_a.choose(list(player_a.history))
        move_b = player_b.choose(list(player_b.history))
        
        if beats(move_a, move_b): wa += 1
        elif beats(move_b, move_a): wb += 1
        else: draws += 1
            
        player_a.update(move_a, move_b)
        player_b.update(move_b, move_a)
        player_a.history.append((move_a, move_b))
        player_b.history.append((move_b, move_a))
        
    return wa / n_rounds, wb / n_rounds, draws / n_rounds

def run_matchup(cls_a, cls_b, n_rounds=1000, n_trials=30):
    wr_a_list, wr_b_list, wr_d_list = [], [], []
    
    for trial in range(n_trials):
        random.seed(42 + trial)
        np.random.seed(42 + trial)
        
        pa = cls_a()
        pb = cls_b()
        
        wa, wb, wd = run_match(pa, pb, n_rounds)
        wr_a_list.append(wa)
        wr_b_list.append(wb)
        wr_d_list.append(wd)
        
    return (statistics.mean(wr_a_list), statistics.mean(wr_b_list), statistics.mean(wr_d_list),
            statistics.stdev(wr_a_list), statistics.stdev(wr_b_list))

# ──────────────────────────────────────────
# 3. 메인 실행 (CSV 추출)
# ──────────────────────────────────────────
if __name__ == "__main__":
    N_ROUNDS = 1000
    N_TRIALS = 30
    DRAW_THRESHOLD = 0.01  # 1.0%p 이내의 격차는 무승부로 판정
    
    # 1. 파일 스캔 및 클래스 로드
    target_files = [f"gpt_v{i}.py" for i in range(1, 8)] + \
                   [f"gemini_v{i}.py" for i in range(1, 8)] + \
                   ["claude_v6.py", "claude_v7.py"]
    
    models = {}
    print("[1/3] 모델 로딩 중...")
    for filename in target_files:
        if os.path.exists(filename):
            try:
                cls = load_player_class(filename, filename)
                model_id = filename.replace('.py', '')
                models[model_id] = cls
                print(f"  - 로드 성공: {model_id}")
            except Exception as e:
                print(f"  ! 로드 실패 ({filename}): {e}")
        else:
            print(f"  ! 파일 없음 (스킵): {filename}")

    model_ids = list(models.keys())
    matchups = list(itertools.combinations(model_ids, 2))
    total_matchups = len(matchups)
    
    print(f"\n[2/3] 총 {len(model_ids)}개의 모델, {total_matchups}개의 매치업 리그전을 시작합니다.")
    print(f"      (소요 시간: 약 3~10분. 차 한 잔 드시고 오세요 ☕)\n")

    results = []
    start_time = time.time()
    
    # 2. 리그전 진행
    for idx, (id_a, id_b) in enumerate(matchups):
        cls_a = models[id_a]
        cls_b = models[id_b]
        
        avg_a, avg_b, avg_d, std_a, std_b = run_matchup(cls_a, cls_b, N_ROUNDS, N_TRIALS)
        
        # 승점 계산
        diff = avg_a - avg_b
        if abs(diff) <= DRAW_THRESHOLD:
            pts_a, pts_b = 1, 1
            result_str = "Draw"
        elif diff > 0:
            pts_a, pts_b = 3, 0
            result_str = f"{id_a} Win"
        else:
            pts_a, pts_b = 0, 3
            result_str = f"{id_b} Win"
            
        results.append({
            'Player_A': id_a,
            'Player_B': id_b,
            'WinRate_A': round(avg_a, 4),
            'WinRate_B': round(avg_b, 4),
            'DrawRate': round(avg_d, 4),
            'StdDev_A': round(std_a, 4),
            'StdDev_B': round(std_b, 4),
            'Points_A': pts_a,
            'Points_B': pts_b,
            'Match_Result': result_str
        })
        
        if (idx + 1) % 10 == 0 or (idx + 1) == total_matchups:
            elapsed = time.time() - start_time
            print(f"  진행률: {idx + 1}/{total_matchups} 완료 ({elapsed:.1f}초 경과)")

    # 3. CSV 저장
    csv_filename = "rps_ultimate_league_results.csv"
    print(f"\n[3/3] 리그전 종료. 결과를 {csv_filename}에 저장했습니다.")
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Player_A', 'Player_B', 'WinRate_A', 'WinRate_B', 'DrawRate', 
                      'StdDev_A', 'StdDev_B', 'Points_A', 'Points_B', 'Match_Result']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(results)
        
    print("모든 작업이 완료되었습니다! 🏆 최종 우승자가 누구인지 꼭 알려주세요!")