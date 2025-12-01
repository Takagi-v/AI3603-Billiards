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
        # 可以在这里初始化一些策略参数
        self.num_simulation = 50 # 蒙特卡洛模拟次数 (Member A 负责策略配置)

    def decision(self, balls, my_targets, table):
        """
        Member A 负责的主决策逻辑：
        1. 遍历所有“目标球-袋口”组合
        2. 调用 Member B 的几何解算器 (solve_shot_parameters)
        3. 调用 Member B 的路径检测器 (is_path_clear)
        4. (未来) 调用 Member A 的评分系统进行优选
        """
        
        # 1. 获取合法的目标球
        legal_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not legal_targets:
            # 如果目标球全进了，就打黑8
            legal_targets = ['8']
            
        cue_ball = balls['cue']
        pockets = table.pockets
        
        candidates = [] # 候选击球动作列表

        print(f"[NewAgent] 正在思考... 剩余目标球: {legal_targets}")

        # 2. 遍历所有可能性
        for target_id in legal_targets:
            target_ball = balls[target_id]
            
            for pocket_id, pocket in pockets.items():
                # --- [Member B 工作区] 几何解算 ---
                # 计算把 target_ball 打进 pocket 所需的击球参数
                action = self.solve_shot_parameters(cue_ball, target_ball, pocket)
                
                if action is None:
                    continue # 物理上打不进（比如角度太死）
                
                # --- [Member B 工作区] 路径检测 ---
                # 检查路线上有没有障碍球
                if not self.is_path_clear(cue_ball, target_ball, pocket, balls):
                    continue # 路线被挡住了
                
                # 如果通过了上述检查，这就是一个“候选动作”
                candidates.append(action)

        # 3. 决策 (Member A 的策略核心)
        if not candidates:
            print("[NewAgent] 没有发现必进球，执行防守或随机击球。")
            return self._random_action()
        
        # 目前阶段：简单返回第一个找到的可行解
        # 下一阶段：在这里加入蒙特卡洛模拟 (simulate_shot) 和走位评分
        print(f"[NewAgent] 发现 {len(candidates)} 个可行解，选择第一个执行。")
        return candidates[0]

    def solve_shot_parameters(self, cue_ball, target_ball, pocket):
        """
        [待 Member B 实现] 几何求解器
        
        功能：
            计算将 target_ball 打进 pocket 所需的白球击球参数 (V0, phi, theta)。
            需要计算“幻影球”位置，并反推白球的击球角度。
            
        参数：
            cue_ball: 白球对象
            target_ball: 目标球对象
            pocket: 目标袋口对象
            
        返回：
            dict: {'V0': float, 'phi': float, 'theta': float, 'a': 0, 'b': 0}
            如果物理上无法打进（例如切球角度 > 90度），返回 None
        """
        # 1. 获取位置信息 (x, y 坐标)
        cue_pos = cue_ball.state.rvw[0]
        target_pos = target_ball.state.rvw[0]
        pocket_pos = pocket.center
        
        # 2. 获取球半径 (从球对象参数中获取)
        R = cue_ball.params.R

        # 3. 计算 target 到 pocket 的向量 (击球线)
        # 我们只关心水平面上的向量 (x, y)
        target_to_pocket = pocket_pos - target_pos
        # 忽略 z 轴差异 (虽然通常 z 是一样的)
        target_to_pocket[2] = 0 
        
        dist_target_pocket = np.linalg.norm(target_to_pocket)
        if dist_target_pocket < 1e-6:
            return None # 已经在袋口了?
            
        # 单位向量
        u_tp = target_to_pocket / dist_target_pocket
        
        # 4. 计算 Ghost Ball 位置
        # Ghost Ball 是白球击中目标球瞬间，白球中心应该所在的位置
        # 它位于目标球中心沿 target_to_pocket 反方向延伸 2R 处
        ghost_pos = target_pos - u_tp * (2 * R)
        
        # 5. 计算白球到 Ghost Ball 的向量 (瞄准线)
        cue_to_ghost = ghost_pos - cue_pos
        cue_to_ghost[2] = 0
        
        dist_cue_ghost = np.linalg.norm(cue_to_ghost)
        if dist_cue_ghost < 1e-6:
            # 白球就在 Ghost Ball 位置，直接打? 这种情况极少
            # 简单的处理：沿 u_tp 方向打
            phi = np.degrees(np.arctan2(u_tp[1], u_tp[0]))
        else:
            u_cg = cue_to_ghost / dist_cue_ghost
            
            # 6. 检查切球角度 (Cut Angle)
            # 向量 cue_to_target
            cue_to_target = target_pos - cue_pos
            cue_to_target[2] = 0
            
            # 如果 cue_to_target 和 target_to_pocket 的夹角超过 90 度，则无法直接打进
            # 判断方法：点积
            # 注意：这里要判断的是 "白球 -> 目标球" 方向 与 "目标球 -> 袋口" 方向的夹角
            # 如果这个夹角 > 90度，说明目标球在白球和袋口之间，可以打
            # 如果夹角 < 90度?? 
            # 正确的物理限制：
            # 切球角度是 u_cg (白球行进方向) 和 u_tp (目标球行进方向) 之间的夹角
            # 只要这个角度 < 90 度，理论上就能让目标球沿 u_tp 运动 (只要摩擦力允许)
            # 但如果 Ghost Ball 被 Target Ball 挡住了（即 Ghost Ball 在 Target Ball "后面"），那就打不到 Ghost Ball
            
            # 让我们用更直观的判断：
            # 向量 cue_to_target (v_ct) 与 u_tp 的点积
            # 如果 v_ct · u_tp > 0，说明白球大致在目标球的“后方”，可以向前击打目标球入袋
            # 如果 v_ct · u_tp < 0，说明白球在目标球的“前方”，需要回打(不可能)
            
            # 更精确的判断：检查 Ghost Ball 是否可达
            # Ghost Ball 必须对白球可见，且不能与 Target Ball 重叠（当然不会，它是虚拟的）
            # 关键是：白球去 Ghost Ball 的路径上，不能先碰到 Target Ball
            # 这意味着：cue_to_ghost 方向上，Target Ball 不能在中间
            # 但实际上，Ghost Ball 就紧贴着 Target Ball。
            # 只要 angle(cue_to_ghost, u_tp) < 90度，就可以。
            # 实际上是 angle(u_cg, u_tp)
            
            dot_prod = np.dot(u_cg, u_tp)
            # 限制在 -1 到 1 之间以防浮点误差
            dot_prod = np.clip(dot_prod, -1.0, 1.0)
            cut_angle = np.degrees(np.arccos(dot_prod))
            
            if cut_angle > 80: # 留一点余量，超过80度很难打
                return None
            
            phi = np.degrees(np.arctan2(u_cg[1], u_cg[0]))

        # 规范化 phi 到 [0, 360)
        phi = phi % 360
        
        # 7. 设定其他参数
        # 简单的力度策略：距离越远力度越大
        # 基础力度 1.5，每米增加 1.0
        # 总距离 = 白球到Ghost + Ghost到袋口 (近似 Target到袋口)
        total_dist = dist_cue_ghost + dist_target_pocket
        V0 = 1.5 + total_dist * 1.5
        V0 = np.clip(V0, 0.5, 8.0) # 限制在允许范围内
        
        # theta, a, b 设为默认值
        theta = 0.0 # 平击
        a = 0.0
        b = 0.0
        
        return {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}

    def is_path_clear(self, cue_ball, target_ball, pocket, balls):
        """
        [待 Member B 实现] 路径检测器
        
        功能：
            检查两个路径段是否被其他球阻挡：
            1. 白球 -> 幻影球 (Ghost Ball)
            2. 目标球 -> 袋口 (Pocket)
            
        参数：
            cue_ball: 白球对象
            target_ball: 目标球对象
            pocket: 目标袋口对象
            balls: 所有球的状态字典 (用于检查障碍物)
            
        返回：
            bool: True 表示路径通畅，False 表示有阻挡
        """
        R = cue_ball.params.R
        
        # 关键点坐标
        cue_pos = cue_ball.state.rvw[0]
        target_pos = target_ball.state.rvw[0]
        pocket_pos = pocket.center
        
        # 计算 Ghost Ball 位置
        target_to_pocket = pocket_pos - target_pos
        target_to_pocket[2] = 0
        dist_tp = np.linalg.norm(target_to_pocket)
        if dist_tp < 1e-6: return True # 应该不会发生
        u_tp = target_to_pocket / dist_tp
        ghost_pos = target_pos - u_tp * (2 * R)
        
        # 定义线段点距离函数
        def point_line_segment_distance(px, py, x1, y1, x2, y2):
            # 计算点 (px, py) 到线段 (x1, y1)-(x2, y2) 的最短距离
            # 向量 AB
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                return math.sqrt((px - x1)**2 + (py - y1)**2)

            # 投影参数 t
            t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
            
            # 限制 t 在 [0, 1]
            t = max(0, min(1, t))
            
            # 最近点
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            
            return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

        # 检查所有球
        for ball_id, ball in balls.items():
            if ball.state.s == 4: # 已进袋的球忽略
                continue
            if ball_id == cue_ball.id or ball_id == target_ball.id:
                continue
            
            pos = ball.state.rvw[0]
            bx, by = pos[0], pos[1]
            
            # 1. 检查路径: 白球 -> Ghost Ball
            # 起点: cue_pos, 终点: ghost_pos
            # 注意：这里的终点是 ghost_pos。Ghost Ball 本身与 Target Ball 相切。
            # 我们的检测应该稍微避开 Target Ball，否则会误报 Target Ball 阻挡
            # 但这里 loop 中已经排除了 target_ball，所以可以直接测
            
            dist1 = point_line_segment_distance(bx, by, cue_pos[0], cue_pos[1], ghost_pos[0], ghost_pos[1])
            if dist1 < 2 * R: # 障碍球中心到路径距离小于 2R，说明会碰撞
                return False
                
            # 2. 检查路径: 目标球 -> 袋口
            # 起点: target_pos, 终点: pocket_pos
            dist2 = point_line_segment_distance(bx, by, target_pos[0], target_pos[1], pocket_pos[0], pocket_pos[1])
            if dist2 < 2 * R:
                return False
                
        return True