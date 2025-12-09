"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...]
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8）, -30（首球/碰库犯规）
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue']
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2:  # 只有白球和8号球时不算犯规
             foul_first_hit = True
    else:
        remaining_own_before = [bid for bid in player_targets if last_state[bid].state.s != 4]
        opponent_plus_eight = [bid for bid in last_state.keys() if bid not in player_targets and bid not in ['cue']]
        if ('8' not in opponent_plus_eight):
            opponent_plus_eight.append('8')
            
        if len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100 if is_targeting_eight_ball_legally else -150
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用 pooltool 物理引擎 (世界A)
                    pt.simulate(shot, inplace=True)
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

class NewAgent(Agent):
    """
    Member A (架构师) & Member B (物理学家) 的合作结晶
    """
    
    def __init__(self):
        super().__init__()
        self.num_simulation = 20 # 蒙特卡洛模拟次数 (降低一点以保证速度)
        # 继承环境的噪声设置用于模拟
        self.noise_std = {
            'V0': 0.1, 'phi': 0.1, 'theta': 0.1, 'a': 0.003, 'b': 0.003
        }

    def decision(self, balls, my_targets, table):
        """
        Member A 负责的主决策逻辑：
        1. 遍历所有“目标球-袋口”组合
        2. 调用 Member B 的几何解算器 (solve_shot_parameters)
        3. 调用 Member B 的路径检测器 (is_path_clear)
        4. [NEW] 调用 Member A 的蒙特卡洛模拟 (simulate_shot) 进行评分
        """
        
        # 1. 获取合法的目标球
        legal_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not legal_targets:
            legal_targets = ['8']
            
        cue_ball = balls['cue']
        pockets = table.pockets
        
        best_action = None
        best_score = -float('inf')
        
        candidates_count = 0

        print(f"[NewAgent] 正在思考... 剩余目标球: {legal_targets}")

        # 2. 遍历所有可能性
        for target_id in legal_targets:
            target_ball = balls[target_id]
            
            for pocket_id, pocket in pockets.items():
                # --- [Member B] 几何解算 ---
                solutions = self.solve_shot_parameters(cue_ball, target_ball, pocket, balls, table)
                if not solutions: continue 
                
                # 路径检测已内置于 solve_shot_parameters
                
                candidates_count += 1
                
                # --- [Member A] 蒙特卡洛模拟与评分 ---
                for sol in solutions:
                    base_phi = sol['phi']
                    shot_type = sol['type']
                    
                    # 估算力度 V0 (简单启发式)
                    # 直球力度适中，翻袋/踢球需要更大力度
                    dist_ct = np.linalg.norm(target_ball.state.rvw[0] - cue_ball.state.rvw[0])
                    base_v0 = np.clip(1.5 + dist_ct * 1.5, 1.0, 5.0)
                    if shot_type != 'Direct':
                        base_v0 *= 1.2 # 增加力度以补偿库边损失
                    
                    # 构造基础动作
                    base_action = {
                        'V0': base_v0,
                        'phi': base_phi,
                        'theta': 0,
                        'a': 0,
                        'b': 0,
                        'shot_type': shot_type # 记录类型以便调试
                    }
                    
                    # [改进] 增加角度微调，对抗物理误差（如Throw效应）
                    v0_choices = [base_v0 * 0.9, base_v0 * 1.1]
                    phi_choices = [base_phi - 0.5, base_phi, base_phi + 0.5]
                    
                    for v0 in v0_choices:
                        if v0 < 0.5 or v0 > 8.0: continue
                        for phi in phi_choices:
                            test_action = base_action.copy()
                            test_action['V0'] = v0
                            test_action['phi'] = phi
                            
                            # 模拟 N 次
                            success_rate, avg_score = self.simulate_shot(test_action, balls, table, target_id, my_targets)
                            
                            # 评分公式
                            final_score = success_rate * 100 + avg_score * 0.5
                            if success_rate < 0.3:
                                final_score -= 50
                            
                            # Debug: 如果发现一个还不错的球，打印一下
                            if success_rate > 0.2:
                                print(f"  > 候选: Target={target_id}, Type={shot_type}, Pocket={pocket_id}, V0={v0:.1f}, phi={phi:.1f}, WinRate={success_rate:.2f}, Score={final_score:.1f}")
                            
                            if final_score > best_score:
                                best_score = final_score
                                best_action = test_action

        # 3. 决策
        if best_action is None:
            print("[NewAgent] 没有发现可靠的进攻机会，执行随机防守。")
            return self._random_action()
        
        print(f"[NewAgent] 评估了 {candidates_count} 个球路，选择最佳方案 (得分: {best_score:.1f})")
        return best_action

    def simulate_shot(self, action, balls, table, target_id, my_targets):
        """
        [Member A 实现] 蒙特卡洛模拟
        
        在脑海中模拟多次击球，返回成功率和平均得分。
        [升级] 加入了对白球洗袋和未碰库犯规的检测与惩罚。
        """
        success_count = 0
        total_score = 0
        
        # 创建模拟环境的基础对象
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        
        for _ in range(self.num_simulation):
            # 1. 复制当前球状态
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # 2. 添加噪声
            noisy_action = {
                'V0': np.clip(action['V0'] + np.random.normal(0, self.noise_std['V0']), 0.5, 8.0),
                'phi': (action['phi'] + np.random.normal(0, self.noise_std['phi'])) % 360,
                'theta': np.clip(action['theta'] + np.random.normal(0, self.noise_std['theta']), 0, 90),
                'a': np.clip(action['a'] + np.random.normal(0, self.noise_std['a']), -0.5, 0.5),
                'b': np.clip(action['b'] + np.random.normal(0, self.noise_std['b']), -0.5, 0.5)
            }
            
            # 3. 物理模拟
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            shot.cue.set_state(**noisy_action)
            pt.simulate(shot, inplace=True)
            
            # 4. 分析结果
            new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and sim_balls[bid].state.s != 4]
            cue_pocketed = 'cue' in new_pocketed
            eight_pocketed = '8' in new_pocketed
            
            # 分析碰撞事件 (检测碰库)
            cue_hit_cushion = False
            target_hit_cushion = False
            first_contact_ball_id = None
            
            for e in shot.events:
                et = str(e.event_type).lower()
                ids = list(e.ids) if hasattr(e, 'ids') else []
                
                # 记录首球
                if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                    other_ids = [i for i in ids if i != 'cue']
                    if other_ids and first_contact_ball_id is None:
                        first_contact_ball_id = other_ids[0]
                
                # 记录碰库
                if 'cushion' in et:
                    if 'cue' in ids: cue_hit_cushion = True
                    if first_contact_ball_id is not None and first_contact_ball_id in ids:
                        target_hit_cushion = True

            # 5. 评分逻辑
            is_success = False
            
            # 判定进球成功
            if target_id in new_pocketed:
                if not cue_pocketed:
                    if not eight_pocketed or target_id == '8':
                        is_success = True
            
            if is_success:
                success_count += 1
                # 基础分 +100
                shot_score = 100
                
                # 走位评分：白球离中心越近越好（避免贴库）
                cue_pos = sim_balls['cue'].state.rvw[0]
                dist_to_center = np.linalg.norm(cue_pos[:2] - np.array([table.w/2, table.l/2]))
                shot_score += (1.0 - dist_to_center) * 10 # 走位分权重
                
                total_score += shot_score
            else:
                # 失败惩罚
                if cue_pocketed:
                    total_score -= 500 # 白球洗袋：重罚
                elif eight_pocketed and target_id != '8':
                    total_score -= 1000 # 误打黑8：直接判负，极刑
                else:
                    # 没进球，检查是否犯规
                    foul = False
                    
                    # 1. 空杆犯规
                    if first_contact_ball_id is None:
                        foul = True
                    
                    # 2. 未碰库犯规 (如果没进球，且白球和目标球都没碰库)
                    if not new_pocketed and not (cue_hit_cushion or target_hit_cushion):
                        foul = True
                        
                    if foul:
                        total_score -= 50 # 犯规惩罚
                    else:
                        total_score += 0 # 正常防守/失误，不扣分也不得分
                
        success_rate = success_count / self.num_simulation
        avg_score = total_score / self.num_simulation if self.num_simulation > 0 else 0
        
        return success_rate, avg_score

    def solve_shot_parameters(self, cue_ball, target_ball, pocket, balls=None, table=None):
        """
        [Member B 实现] 几何求解器 (重构版 - 严格物理校验)
        
        功能：
            计算击球角度 phi。
            支持：Direct, Kick, Bank, Kick-Bank。
            
        参数：
            cue_ball, target_ball, pocket: 对象
            balls: 所有球状态
            table: 球桌对象
            
        返回：
            list: 包含所有可行解的列表，每个元素为字典 {'phi': float, 'type': str, ...}
                  如果无解则返回空列表 []
        """
        if balls is None or table is None:
            return []

        import itertools

        solutions = []
        R = cue_ball.params.R
        cue_pos = cue_ball.state.rvw[0]
        target_pos = target_ball.state.rvw[0]
        pocket_pos = pocket.center
        
        # 定义库边
        rails = [
            {'name': 'left',   'val': 0,       'axis': 0, 'limit': (0, table.l)}, # x=0, y in [0, l]
            {'name': 'right',  'val': table.w, 'axis': 0, 'limit': (0, table.l)}, # x=w, y in [0, l]
            {'name': 'bottom', 'val': 0,       'axis': 1, 'limit': (0, table.w)}, # y=0, x in [0, w]
            {'name': 'top',    'val': table.l, 'axis': 1, 'limit': (0, table.w)}  # y=l, x in [0, w]
        ]
        
        # 辅助函数：检测线段是否无障碍
        def is_segment_clear(p1, p2, ignore_target=True):
            vec = p2 - p1
            length = np.linalg.norm(vec)
            if length < 1e-6: return True
            u = vec / length
            
            for bid, ball in balls.items():
                if ball.state.s == 4: continue
                if bid == cue_ball.id: continue
                if ignore_target and bid == target_ball.id: continue
                
                pos = ball.state.rvw[0]
                ap = pos - p1
                proj = np.dot(ap, u)
                if proj < 0 or proj > length:
                    dist = min(np.linalg.norm(pos - p1), np.linalg.norm(pos - p2))
                else:
                    dist = np.linalg.norm(ap - proj * u)
                
                if dist < 2 * R:
                    return False
            return True

        # 辅助函数：镜像点
        def get_mirror(point, rail):
            p = point.copy()
            p[rail['axis']] = 2 * rail['val'] - p[rail['axis']]
            return p
            
        # 辅助函数：计算多库路径点
        # start_pos -> [rail_1, rail_2, ...] -> end_pos
        # 返回: [start, p1, p2, ..., end] 或 None
        def get_cushion_path(start_pos, end_pos, rail_sequence):
            # 1. 从后往前生成镜像目标点
            mirrored_targets = []
            current_target = end_pos
            for rail in reversed(rail_sequence):
                current_target = get_mirror(current_target, rail)
                mirrored_targets.append(current_target)
            
            # 2. 从前往后计算交点
            targets_to_aim = mirrored_targets[::-1] # [First_Mirror, Second_Mirror, ..., End]
            path_points = [start_pos]
            current_p = start_pos
            
            for i, rail in enumerate(rail_sequence):
                target_to_aim = targets_to_aim[i]
                
                vec = target_to_aim - current_p
                vec[2] = 0
                
                if abs(vec[rail['axis']]) < 1e-6: return None
                t = (rail['val'] - current_p[rail['axis']]) / vec[rail['axis']]
                
                if t <= 1e-4: return None # 必须向前
                
                p_intersect = current_p + t * vec
                p_intersect[2] = 0
                
                # 检查交点是否在库边范围内
                other_axis = 1 - rail['axis']
                limit_min, limit_max = rail['limit']
                if not (limit_min <= p_intersect[other_axis] <= limit_max):
                    return None
                    
                path_points.append(p_intersect)
                current_p = p_intersect
            
            path_points.append(end_pos)
            return path_points

        # 辅助函数：尝试从 start_pos 击打到 end_pos (Ghost Ball)
        # 支持直球和踢球 (Kick)
        # 返回: list of dict {'method', 'phi', 'cushions', 'seq', 'u_arrival'}
        def solve_cue_path(start_pos, end_pos, target_ball_pos):
            sub_solutions = []
            
            # 1. 直打 (Direct Hit to Ghost)
            vec = end_pos - start_pos
            vec[2] = 0
            dist = np.linalg.norm(vec)
            if dist > 1e-6:
                u = vec / dist
                if is_segment_clear(start_pos, end_pos, ignore_target=True):
                     phi = np.degrees(np.arctan2(u[1], u[0])) % 360
                     sub_solutions.append({
                         'method': 'Direct',
                         'phi': phi,
                         'cushions': 0,
                         'seq': [],
                         'u_arrival': u # 直打到达方向就是 vec 方向
                     })

            # 2. 踢球 (Kick Hit to Ghost)
            max_kick_cushions = 1
            for n in range(1, max_kick_cushions + 1):
                all_seqs = []
                for seq in itertools.product(rails, repeat=n):
                    valid = True
                    for i in range(len(seq)-1):
                        if seq[i]['name'] == seq[i+1]['name']: valid = False; break
                    if valid: all_seqs.append(seq)
                
                for seq in all_seqs:
                    path_points = get_cushion_path(start_pos, end_pos, seq)
                    if path_points:
                        # 路径检查
                        path_clear = True
                        for i in range(len(path_points)-1):
                            p1, p2 = path_points[i], path_points[i+1]
                            # 检测除 Target 外的障碍
                            if not is_segment_clear(p1, p2, ignore_target=True):
                                path_clear = False; break
                            
                            # 单独检测 Target 阻挡 (非最后一段)
                            if i < len(path_points)-2:
                                vec_seg = p2 - p1
                                len_seg = np.linalg.norm(vec_seg)
                                u_seg = vec_seg / len_seg
                                ap = target_ball_pos - p1
                                proj = np.dot(ap, u_seg)
                                if 0 < proj < len_seg:
                                    dist = np.linalg.norm(ap - proj * u_seg)
                                    if dist < 2 * R:
                                        path_clear = False; break
                        
                        if path_clear:
                             # 计算初始角度
                             p1 = path_points[1]
                             vec_shot = p1 - start_pos
                             phi = np.degrees(np.arctan2(vec_shot[1], vec_shot[0])) % 360
                             
                             # 计算到达 Ghost 时的方向 (最后一段: Last_Collision -> Ghost)
                             p_last_coll = path_points[-2]
                             vec_arr = end_pos - p_last_coll
                             vec_arr[2] = 0
                             u_arrival = vec_arr / np.linalg.norm(vec_arr)
                             
                             sub_solutions.append({
                                 'method': 'Kick',
                                 'phi': phi,
                                 'cushions': n,
                                 'seq': [r['name'] for r in seq],
                                 'u_arrival': u_arrival
                             })
            return sub_solutions

        # ==========================================
        # 主流程：两阶段求解
        # ==========================================
        
        # 阶段 A: 确定 Target -> Pocket 的路径 (Direct 或 Bank)
        target_strategies = []
        
        # A1. Target Direct
        target_to_pocket = pocket_pos - target_pos
        target_to_pocket[2] = 0
        u_tp = target_to_pocket / np.linalg.norm(target_to_pocket)
        ghost_direct = target_pos - u_tp * (2 * R)
        
        if is_segment_clear(target_pos, pocket_pos):
             target_strategies.append({
                 'type': 'Direct',
                 'ghost': ghost_direct,
                 'u_target_out': u_tp, # 目标球被击打后的期望方向
                 'seq': []
             })
             
        # A2. Target Bank
        max_bank_cushions = 1
        for n in range(1, max_bank_cushions + 1):
            all_seqs = []
            for seq in itertools.product(rails, repeat=n):
                valid = True
                for i in range(len(seq)-1):
                    if seq[i]['name'] == seq[i+1]['name']: valid = False; break
                if valid: all_seqs.append(seq)
            
            for seq in all_seqs:
                path_points = get_cushion_path(target_pos, pocket_pos, seq)
                if path_points:
                    path_clear = True
                    for i in range(len(path_points)-1):
                        if not is_segment_clear(path_points[i], path_points[i+1]):
                            path_clear = False; break
                    
                    if path_clear:
                        p1 = path_points[1]
                        vec_t = p1 - target_pos
                        vec_t[2] = 0
                        u_t = vec_t / np.linalg.norm(vec_t)
                        ghost_bank = target_pos - u_t * (2 * R)
                        
                        target_strategies.append({
                            'type': 'Bank',
                            'ghost': ghost_bank,
                            'u_target_out': u_t,
                            'seq': [r['name'] for r in seq]
                        })

        # 阶段 B: 求解 Cue -> Ghost
        seen_phis = set() # 去重集合
        
        for strat in target_strategies:
            ghost = strat['ghost']
            u_target_out = strat['u_target_out'] # 目标球应去的方向
            
            cue_paths = solve_cue_path(cue_pos, ghost, target_pos)
            
            for cp in cue_paths:
                # 关键验证：切球角度 (Cut Angle Check)
                # 白球到达 Ghost 时，能否产生 u_target_out 的力？
                # 白球到达方向 u_arrival
                # 切球角 = arccos(u_arrival dot u_target_out)
                # 注意：u_arrival 是白球行进方向。
                # 碰撞点是 Ghost (Target - 2R * u_target_out)。
                # 碰撞法线 (Normal) = Target - Ghost = 2R * u_target_out. 即 u_target_out.
                # 实际上，只要 u_arrival 和 u_target_out 的夹角 < 90 度，
                # 理论上就能推动目标球向前（只要不是反向撞击）。
                # 但为了保证动量传递效率，一般限制在 80 度以内。
                
                cut_angle = np.degrees(np.arccos(np.clip(np.dot(cp['u_arrival'], u_target_out), -1, 1)))
                
                if cut_angle < 80:
                    # 构造解
                    final_type = strat['type']
                    if cp['method'] == 'Kick':
                        if final_type == 'Direct': final_type = 'Kick'
                        elif final_type == 'Bank': final_type = 'Kick-Bank'
                    
                    # 去重：基于 phi (保留 2 位小数)
                    phi_key = round(cp['phi'], 2)
                    if phi_key in seen_phis:
                        continue
                    seen_phis.add(phi_key)
                    
                    combined_seq = cp['seq'] + strat['seq']
                    total_cushions = cp['cushions'] + (len(strat['seq']) if strat['type'] == 'Bank' else 0)
                    
                    solutions.append({
                        'type': final_type,
                        'phi': cp['phi'],
                        'cushions': total_cushions,
                        'seq': combined_seq,
                        'cut_angle': cut_angle
                    })

        return solutions

    def is_path_clear(self, cue_ball, target_ball, pocket, balls):
        """
        [Member B 实现] 路径检测器
        """
        R = cue_ball.params.R
        cue_pos = cue_ball.state.rvw[0]
        target_pos = target_ball.state.rvw[0]
        pocket_pos = pocket.center
        
        target_to_pocket = pocket_pos - target_pos
        target_to_pocket[2] = 0
        dist_tp = np.linalg.norm(target_to_pocket)
        if dist_tp < 1e-6: return True
        u_tp = target_to_pocket / dist_tp
        ghost_pos = target_pos - u_tp * (2 * R)
        
        def point_line_segment_distance(px, py, x1, y1, x2, y2):
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                return math.sqrt((px - x1)**2 + (py - y1)**2)
            t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
            t = max(0, min(1, t))
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

        for ball_id, ball in balls.items():
            if ball.state.s == 4: continue
            if ball_id == cue_ball.id or ball_id == target_ball.id: continue
            
            pos = ball.state.rvw[0]
            bx, by = pos[0], pos[1]
            
            if point_line_segment_distance(bx, by, cue_pos[0], cue_pos[1], ghost_pos[0], ghost_pos[1]) < 2 * R:
                return False
            if point_line_segment_distance(bx, by, target_pos[0], target_pos[1], pocket_pos[0], pocket_pos[1]) < 2 * R:
                return False
                
        return True