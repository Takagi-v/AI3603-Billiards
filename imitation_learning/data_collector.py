import os
import sys
import time
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
import contextlib
import random

# 将上级目录添加到 path 以便导入 agent 和 poolenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poolenv import PoolEnv
from agent import NewAgent

# ================= 配置区域 =================
# 采集总局数
TOTAL_GAMES = 200000
# 进程数 (先降低到4个测试，确认能跑通后再增加)
NUM_PROCESSES = 4
# 每个进程负责的局数
GAMES_PER_PROCESS = TOTAL_GAMES // NUM_PROCESSES

# 数据保存路径
DATA_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_SAVE_DIR, exist_ok=True)

# 抑制输出的上下文管理器
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def get_canonical_state(balls, my_targets, opponent_targets, table_w, table_l):
    """
    状态标准化/特征提取函数 (34维特征)
    """
    features = []
    
    # 1. Cue Ball
    cue = balls['cue']
    features.extend([cue.state.rvw[0][0], cue.state.rvw[0][1]])
    
    # 2. Eight Ball
    eight = balls['8']
    if eight.state.s == 4: # 进袋
        features.extend([-1.0, -1.0])
    else:
        features.extend([eight.state.rvw[0][0], eight.state.rvw[0][1]])
        
    pos_map = {}
    for bid, ball in balls.items():
        if bid == 'cue' or bid == '8': continue
        if ball.state.s == 4:
            pos_map[bid] = [-1.0, -1.0]
        else:
            pos_map[bid] = [ball.state.rvw[0][0], ball.state.rvw[0][1]]
            
    solids_features = []
    for i in range(1, 8):
        solids_features.extend(pos_map.get(str(i), [-1.0, -1.0]))
        
    stripes_features = []
    for i in range(9, 16):
        stripes_features.extend(pos_map.get(str(i), [-1.0, -1.0]))
        
    # 确定谁是 My，统一视角：前14维是己方球，后14维是对方球
    is_me_solid = False
    
    # 简单的启发式判断
    original_solids = set([str(i) for i in range(1, 8)])
    original_stripes = set([str(i) for i in range(9, 16)])
    current_my_set = set(my_targets)
    current_opp_set = set(opponent_targets)
    
    if len(current_my_set.intersection(original_solids)) > 0:
        is_me_solid = True
    elif len(current_my_set.intersection(original_stripes)) > 0:
        is_me_solid = False
    else:
        # 只剩8了，看对手
        if len(current_opp_set.intersection(original_solids)) > 0:
            is_me_solid = False 
        elif len(current_opp_set.intersection(original_stripes)) > 0:
            is_me_solid = True 
        else:
            is_me_solid = True
            
    if is_me_solid:
        features.extend(solids_features)
        features.extend(stripes_features)
    else:
        features.extend(stripes_features)
        features.extend(solids_features)
        
    return features


def worker_task(process_id, num_games, shared_counter):
    """
    工作进程：跑 num_games 局游戏，收集数据
    """
    # === 关键优化：重置随机种子 ===
    # 确保每个进程生成的随机序列完全独立
    seed = (int(time.time() * 1000) + process_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    
    filename = os.path.join(DATA_SAVE_DIR, f'data_{process_id}.csv')
    data_buffer = []

    # 每一批写入一次，防止内存爆炸
    BATCH_SIZE = 50
    
    with suppress_stdout():
        # 开启环境噪声 (env.enable_noise = True)
        # 理由：虽然我们学习的是 Agent 的意图，但开启噪声会让每一杆后的球停在略微不同的位置。
        # 这通过物理模拟自然地增加了数据多样性 (Data Augmentation)，让 Agent 见过更多微小的局面变化。
        # 否则 20万局里会有大量重复的完美走位局面。
        env = PoolEnv()
        env.enable_noise = True 
        
        agent_a = NewAgent()
        agent_b = NewAgent() # Self-Play
        
        games_played = 0
        
        while games_played < num_games:
            target_type = random.choice(['solid', 'stripe'])
            env.reset(target_ball=target_type)
            
            while True:
                player_id = env.get_curr_player()
                opponent_id = 'B' if player_id == 'A' else 'A'
                
                # 获取观测
                balls, my_targets, table = env.get_observation(player_id)
                opponent_targets = env.player_targets[opponent_id]
                
                # 获取动作
                current_agent = agent_a if player_id == 'A' else agent_b
                try:
                    action = current_agent.decision(balls, my_targets, table)
                    
                    # === 策略核心 ===
                    # 只要是 Agent 正常思考后的输出，我们都应该学习。
                    # 哪怕实际执行结果不好（被环境噪声干扰），但它的意图是好的。
                    # 我们要学的是“意图”。
                    
                    state_vec = get_canonical_state(balls, my_targets, opponent_targets, table.w, table.l)
                    action_vec = [
                        action['V0'],
                        action['phi'],
                        action['theta'],
                        action['a'],
                        action['b']
                    ]
                    
                    data_buffer.append(state_vec + action_vec)
                    
                except Exception:
                    # 异常情况不记录
                    action = current_agent._random_action()

                # 执行动作
                env.take_shot(action)
                
                done, info = env.get_done()
                if done:
                    break
            
            games_played += 1
            
            # 更新共享计数器 (原子操作)
            with shared_counter.get_lock():
                shared_counter.value += 1
            
            if games_played % BATCH_SIZE == 0:
                _save_buffer(data_buffer, filename)
                data_buffer = []
        
        if data_buffer:
            _save_buffer(data_buffer, filename)
            
    return games_played


def _save_buffer(buffer, filename):
    if not buffer: return
    df = pd.DataFrame(buffer)
    # State: 34 cols, Action: 5 cols. Total 39 cols.
    # header只在第一次写入
    mode = 'a' if os.path.exists(filename) else 'w'
    header = not os.path.exists(filename)
    try:
        df.to_csv(filename, mode=mode, header=header, index=False)
    except:
        pass


def main():
    print(f"=== 模仿学习数据采集启动 ===")
    print(f"目标: {TOTAL_GAMES} 局 | 进程数: {NUM_PROCESSES}")
    print(f"保存: {DATA_SAVE_DIR}")
    print("正在启动进程池 (预计耗时较长)...")
    
    # 共享计数器
    manager = multiprocessing.Manager()
    shared_counter = multiprocessing.Value('i', 0)
    
    start_time = time.time()
    
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        for i in range(NUM_PROCESSES):
            pool.apply_async(worker_task, args=(i, GAMES_PER_PROCESS, shared_counter))
        
        # 主进程显示进度条
        pbar = tqdm(total=TOTAL_GAMES, unit="game", desc="Collecting")
        last_count = 0
        
        while True:
            current_count = shared_counter.value
            delta = current_count - last_count
            if delta > 0:
                pbar.update(delta)
                last_count = current_count
            
            if current_count >= TOTAL_GAMES * 0.999: # 接近完成时稍微等待一下worker退出
                # 简单处理：等到 >= TOTAL_GAMES
                 if current_count >= TOTAL_GAMES:
                     pbar.update(TOTAL_GAMES - pbar.n)
                     break
            
            time.sleep(1.0) # 1秒刷新一次
            
        pbar.close()
        # 等待所有子进程彻底退出 (join)
        pool.close()
        pool.join()
        
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n=== 采集完成 ===")
    print(f"耗时: {duration:.2f}s | 速度: {TOTAL_GAMES / duration:.2f} games/s")
    print(f"数据已保存。现在你可以开始训练模型了。")

if __name__ == '__main__':
    # Linux (AutoDL) 上使用 'fork' 启动最快且内存占用最小
    # 这通常能解决进程卡住起不来的问题
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    main()
