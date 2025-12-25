import os
import sys
import time
import numpy as np
import pandas as pd
import random
import contextlib

# 将上级目录添加到 path 以便导入 agent 和 poolenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poolenv import PoolEnv
from agent import NewAgent

DATA_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_SAVE_DIR, exist_ok=True)

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
    features = []
    cue = balls['cue']
    features.extend([cue.state.rvw[0][0], cue.state.rvw[0][1]])
    
    eight = balls['8']
    if eight.state.s == 4:
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
        
    is_me_solid = False
    original_solids = set([str(i) for i in range(1, 8)])
    original_stripes = set([str(i) for i in range(9, 16)])
    current_my_set = set(my_targets)
    current_opp_set = set(opponent_targets)
    
    if len(current_my_set.intersection(original_solids)) > 0:
        is_me_solid = True
    elif len(current_my_set.intersection(original_stripes)) > 0:
        is_me_solid = False
    else:
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


def _save_buffer(buffer, filename):
    if not buffer: return
    df = pd.DataFrame(buffer)
    # 34 + 5 = 39 columns
    mode = 'a' if os.path.exists(filename) else 'w'
    header = not os.path.exists(filename)
    try:
        df.to_csv(filename, mode=mode, header=header, index=False)
    except:
        pass


def worker_main():
    if len(sys.argv) < 3:
        # Prevent run without args from doing anything
        return

    worker_id = int(sys.argv[1])
    num_games = int(sys.argv[2])
    
    seed = (int(time.time() * 1000) + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    
    filename = os.path.join(DATA_SAVE_DIR, f'data_{worker_id}.csv')
    data_buffer = []
    BATCH_SIZE = 1
    games_played = 0

    with suppress_stdout():
        env = PoolEnv()
        env.enable_noise = True
        
        agent_a = NewAgent()
        agent_b = NewAgent()

        while games_played < num_games:
            target_type = random.choice(['solid', 'stripe'])
            env.reset(target_ball=target_type)
            
            while True:
                player_id = env.get_curr_player()
                opponent_id = 'B' if player_id == 'A' else 'A'
                
                balls, my_targets, table = env.get_observation(player_id)
                opponent_targets = env.player_targets[opponent_id]
                
                current_agent = agent_a if player_id == 'A' else agent_b
                try:
                    action = current_agent.decision(balls, my_targets, table)
                    state_vec = get_canonical_state(balls, my_targets, opponent_targets, table.w, table.l)
                    action_vec = [
                        action['V0'], action['phi'], action['theta'], action['a'], action['b']
                    ]
                    data_buffer.append(state_vec + action_vec)
                except Exception:
                    action = current_agent._random_action()

                env.take_shot(action)
                done, info = env.get_done()
                if done:
                    break
            
            games_played += 1
            
            # 实时汇报进度 (绕过 stdout 抑制)
            msg = f"[Worker {worker_id}] Game {games_played}/{num_games} finished. (Data: {len(data_buffer)})\n"
            sys.__stdout__.write(msg)
            sys.__stdout__.flush()
            
            if games_played % BATCH_SIZE == 0:
                _save_buffer(data_buffer, filename)
                data_buffer = []
        
        if data_buffer:
            _save_buffer(data_buffer, filename)
            
    # Touch done file
    done_file = os.path.join(DATA_SAVE_DIR, f'worker_{worker_id}.done')
    with open(done_file, 'w') as f:
        f.write('done')

if __name__ == '__main__':
    worker_main()
