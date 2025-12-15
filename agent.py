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
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
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
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
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
                    
                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
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

class VirtualState:
    def __init__(self, pos):
        self.rvw = np.array([np.array(pos), np.zeros(3), np.zeros(3)])
        self.s = 1

class VirtualBall:
    def __init__(self, ball_id, pos, R=0.028575):
        self.id = ball_id
        self.state = VirtualState(pos)
        self.params = type('Params', (), {'R': R})()

class NewAgent(Agent):
    """
    重构后的 NewAgent - 模块化决策架构
    
    架构:
    1. solve_shot_parameters - 纯几何求解
    2. simulate_and_score - 蒙特卡洛模拟+评分
    3. _evaluate_position - 走位评分
    4. evaluate_attack_options - 评估所有进攻选项
    5. evaluate_defense_options - 防守评估（复用进攻评估器模拟对手）
    6. decision - 精简的主入口
    """
    
    def __init__(self):
        super().__init__()
        # 蒙特卡洛模拟参数
        self.num_simulation = 15  # 精细模拟次数
        self.num_quick_simulation = 5  # 快速筛选模拟次数
        self.top_k_candidates = 5  # 精细模拟的候选数量
        
        # 噪声参数（与环境一致）
        self.noise_std = {
            'V0': 0.1, 'phi': 0.1, 'theta': 0.1, 'a': 0.003, 'b': 0.003
        }
        
        # 决策参数
        self.base_attack_threshold = 50  # 进攻阈值基准（已提高）
        self.default_v0 = 2.5  # 默认力度（后续可优化）
        
        print("[NewAgent] 模块化架构初始化完成")

    # ==================== 主决策入口 ====================
    
    def decision(self, balls, my_targets, table):
        """
        主决策入口 - 精简版
        
        流程:
        0. 检测是否为开球局面
        1. 评估所有进攻选项
        2. 如果最高分 >= 阈值: 进攻
        3. 否则: 评估防守选项，选择最佳防守
        """
        # 1. 获取合法目标球
        legal_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not legal_targets:
            legal_targets = ['8']
        
        cue_pos = balls['cue'].state.rvw[0]
        print(f"\n{'='*60}")
        print(f"[NewAgent] 正在思考... 剩余目标球: {legal_targets}")
        print(f"  白球位置: ({cue_pos[0]:.3f}, {cue_pos[1]:.3f})")
        
        # 0. 检测是否为开球局面
        if self._is_break_shot(balls, table):
            print(f"\n  === 检测到开球局面 ===")
            action = self._break_shot_action(balls, table)
            print(f"\n  >>> 决策: 大力开球!")
            print(f"      动作: V0={action['V0']:.2f}, phi={action['phi']:.1f}°")
            print(f"{'='*60}\n")
            return action
        
        # 2. 评估所有进攻选项
        print(f"\n  === 进攻评估 ===")
        attack_options = self.evaluate_attack_options(balls, legal_targets, table)
        
        # 3. 动态阈值
        remaining_own = len(legal_targets)
        attack_threshold = self._get_attack_threshold(remaining_own)
        
        # 筛选符合条件的进攻方案：成功率 >= 60% 斩杀线
        MIN_SUCCESS_RATE = 0.5  # 成功率斩杀线
        valid_attacks = [opt for opt in attack_options if opt['success_rate'] >= MIN_SUCCESS_RATE]
        
        best_attack_score = valid_attacks[0]['final_score'] if valid_attacks else -100
        best_attack = valid_attacks[0] if valid_attacks else None
        
        if valid_attacks:
            print(f"\n  符合成功率要求(>={MIN_SUCCESS_RATE:.0%})的方案: {len(valid_attacks)}个")
            print(f"  最高分: {best_attack_score:.1f}, 阈值: {attack_threshold:.1f}")
        else:
            print(f"\n  没有成功率>={MIN_SUCCESS_RATE:.0%}的方案，切换防守")
        
        # 4. 决策
        if best_attack and best_attack_score >= attack_threshold:
            action = best_attack['action']
            cushions = best_attack.get('cushions', 0)
            cushion_str = f"{cushions}库" if cushions > 0 else "直球"
            print(f"\n  >>> 决策: 进攻!")
            print(f"      目标: {best_attack['target_id']} -> {best_attack['pocket_id']}")
            print(f"      类型: {best_attack['shot_type']} ({cushion_str})")
            print(f"      动作: V0={action['V0']:.2f}, phi={action['phi']:.1f}°")
            print(f"      成功率: {best_attack['success_rate']:.0%}, 走位分: {best_attack['position_score']:.1f}, 总分: {best_attack_score:.1f}")
            print(f"{'='*60}\n")
            return action
        else:
            # 5. 防守模式
            print(f"\n  进攻分数不足或无高成功率方案，切换防守模式...")
            print(f"\n  === 防守评估 ===")
            
            opp_targets = self._get_opponent_targets(my_targets)
            defense_options = self.evaluate_defense_options(balls, legal_targets, opp_targets, table)
            
            if defense_options and defense_options[0]['defense_score'] > -50:
                best = defense_options[0]
                action = best['action']
                print(f"\n  >>> 决策: 防守!")
                print(f"      策略: {best.get('strategy', 'Unknown')}")
                print(f"      动作: V0={action['V0']:.2f}, phi={action['phi']:.1f}°")
                print(f"      预测白球停点: ({best.get('predicted_stop', [0,0,0])[0]:.2f}, {best.get('predicted_stop', [0,0,0])[1]:.2f})")
                print(f"      对手最佳: {best.get('opp_best_score', 0):.1f}, 防守得分: {best['defense_score']:.1f}")
                print(f"{'='*60}\n")
                return action
            else:
                # Fallback: 轻打向目标球（保守防守）
                print(f"\n  >>> 决策: 保守防守 (轻打目标球)")
                action = self._fallback_defense(balls, legal_targets[0], table)
                print(f"      动作: V0={action['V0']:.2f}, phi={action['phi']:.1f}°")
                print(f"{'='*60}\n")
                return action

    def _get_attack_threshold(self, remaining_own):
        """动态进攻阈值 - 已提高"""
        base = self.base_attack_threshold  # 基础值已改为 50
        if remaining_own <= 1:
            return base * 0.6  # 最后一球/黑8，激进
        elif remaining_own <= 2:
            return base * 0.8  # 快清台，激进
        elif remaining_own >= 5:
            return base * 1.2  # 还早，保守
        return base

    def _fallback_defense(self, balls, target_id, table):
        """
        保守防守 fallback：轻打向目标球
        当没有找到任何防守方案时使用
        """
        cue_pos = balls['cue'].state.rvw[0]
        target_pos = balls[target_id].state.rvw[0]
        
        # 计算击球角度
        direction = target_pos - cue_pos
        phi = np.degrees(np.arctan2(direction[1], direction[0])) % 360
        
        # 轻打，让白球和目标球都不走太远
        return {
            'V0': 1.5,  # 低力度
            'phi': phi,
            'theta': 0,
            'a': 0,
            'b': 0
        }

    def _get_opponent_targets(self, my_targets):
        """获取对手目标球"""
        all_solids = [str(i) for i in range(1, 8)]
        all_stripes = [str(i) for i in range(9, 16)]
        
        if my_targets[0] in all_solids or (my_targets == ['8'] and '1' in all_solids):
            return all_stripes
        else:
            return all_solids

    def _is_break_shot(self, balls, table):
        """
        检测是否为开球局面
        
        判断条件：
        1. 所有目标球（1-15）都在场上
        2. 目标球的位置分布集中在球堆区域（台球桌上半部分的中心区域）
        """
        # 检查所有目标球是否都在场上
        target_balls = [str(i) for i in range(1, 16)]
        all_on_table = all(
            bid in balls and balls[bid].state.s != 4 
            for bid in target_balls
        )
        if not all_on_table:
            return False
        
        # 检查球堆是否集中（计算位置标准差）
        positions = []
        for bid in target_balls:
            if bid in balls and balls[bid].state.s != 4:
                pos = balls[bid].state.rvw[0]
                positions.append(pos[:2])
        
        if len(positions) < 10:
            return False
        
        positions = np.array(positions)
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        avg_dist = np.mean(distances)
        
        # 如果平均距离小于 0.15m（球堆紧凑），认为是开球局面
        # 正常散开后平均距离会 > 0.3m
        return avg_dist < 0.15

    def _break_shot_action(self, balls, table):
        """
        生成开球动作
        
        策略：大力击向球堆中心
        """
        cue_pos = balls['cue'].state.rvw[0]
        
        # 计算球堆中心（通常在台球桌上方 1/3 处）
        target_balls = [str(i) for i in range(1, 16)]
        positions = []
        for bid in target_balls:
            if bid in balls and balls[bid].state.s != 4:
                positions.append(balls[bid].state.rvw[0][:2])
        
        if positions:
            rack_center = np.mean(positions, axis=0)
        else:
            # 默认球堆位置
            rack_center = np.array([table.w / 2, table.l * 0.75])
        
        # 计算击球角度
        direction = rack_center - cue_pos[:2]
        phi = np.degrees(np.arctan2(direction[1], direction[0])) % 360
        
        # 大力开球
        return {
            'V0': 7.5,  # 大力
            'phi': phi,
            'theta': 0,
            'a': 0,
            'b': 0
        }

    # ==================== 进攻评估 ====================
    
    def evaluate_attack_options(self, balls, legal_targets, table):
        """
        评估所有进攻选项
        
        流程:
        1. 遍历所有 (目标球, 袋口) 组合
        2. 几何求解获取候选角度
        3. 快速筛选 Top-K
        4. 对 Top-K 精细模拟评分
        
        返回:
            list: 按 final_score 降序排列的方案列表
        """
        cue_ball = balls['cue']
        pockets = table.pockets
        
        all_candidates = []
        
        # Step 1 & 2: 遍历并几何求解
        for target_id in legal_targets:
            target_ball = balls[target_id]
            
            for pocket_id, pocket in pockets.items():
                solutions = self.solve_shot_parameters(cue_ball, target_ball, pocket, balls, table)
                
                for sol in solutions:
                    # 根据距离估算力度
                    cue_pos = cue_ball.state.rvw[0]
                    target_pos = target_ball.state.rvw[0]
                    pocket_pos = pocket.center
                    
                    dist_cue_to_target = np.linalg.norm(target_pos - cue_pos)
                    dist_target_to_pocket = np.linalg.norm(pocket_pos - target_pos)
                    total_dist = dist_cue_to_target + dist_target_to_pocket
                    
                    # 力度估算公式：增加基础力度
                    # 台球桌长约2m，需要较大力度
                    base_v0 = 2.5 + total_dist * 2.0  # 提高基础值和系数
                    cushions = sol.get('cushions', 0)
                    if cushions > 0:
                        base_v0 *= (1.0 + cushions * 0.2)  # 每颗星增加 20% 力度
                    
                    # 限制在合理范围
                    estimated_v0 = np.clip(base_v0, 2.5, 7.0)
                    
                    # 构造动作
                    action = {
                        'V0': estimated_v0,
                        'phi': sol['phi'],
                        'theta': 0,
                        'a': 0,
                        'b': 0
                    }
                    
                    # 快速评分（基于几何特征）
                    quick_score = self._quick_score(sol)
                    
                    all_candidates.append({
                        'action': action,
                        'target_id': target_id,
                        'pocket_id': pocket_id,
                        'shot_type': sol['type'],
                        'cut_angle': sol.get('cut_angle', 0),
                        'cushions': sol.get('cushions', 0),
                        'quick_score': quick_score,
                        'solution': sol
                    })
        
        if not all_candidates:
            return []
        
        # Step 3: 快速筛选 Top-K（按几何评分）
        all_candidates.sort(key=lambda x: -x['quick_score'])
        
        # ==================== 两阶段筛选 ====================
        
        # 第一阶段：对 Top-15 进行快速模拟（3次），筛选出有进球可能的方案
        stage1_candidates = all_candidates[:15]  # 扩大初筛范围
        stage1_results = []
        
        print(f"  [阶段1] 快速模拟 {len(stage1_candidates)} 个候选...")
        
        for candidate in stage1_candidates:
            # 快速模拟（只模拟3次）
            quick_result = self._quick_simulate(
                candidate['action'],
                balls,
                table,
                candidate['target_id'],
                legal_targets
            )
            candidate['quick_success_rate'] = quick_result['success_rate']
            candidate['quick_foul_rate'] = quick_result['foul_rate']
            
            # 只保留有进球可能的方案（成功率 > 0 且无致命犯规）
            if quick_result['success_rate'] > 0 or quick_result['foul_rate'] < 0.5:
                stage1_results.append(candidate)
        
        print(f"  [阶段1] 有效方案: {len(stage1_results)} 个")
        
        if not stage1_results:
            # 如果没有有效方案，退而求其次选择原始 Top-5
            stage1_results = stage1_candidates[:5]
        
        # 按快速成功率排序，取 Top-5 进入第二阶段
        stage1_results.sort(key=lambda x: (-x['quick_success_rate'], -x['quick_score']))
        stage2_candidates = stage1_results[:self.top_k_candidates]
        
        # 第二阶段：对 Top-5 进行精细模拟（15次）
        print(f"  [阶段2] 精细模拟 Top-{len(stage2_candidates)} 候选:")
        
        scored_candidates = []
        for idx, candidate in enumerate(stage2_candidates):
            score_result = self.simulate_and_score(
                candidate['action'],
                balls,
                table,
                candidate['target_id'],
                legal_targets
            )
            
            candidate.update(score_result)
            scored_candidates.append(candidate)
            
            # 详细调试输出：每个候选方案
            action = candidate['action']
            cushions = candidate.get('cushions', 0)
            cushion_str = f"{cushions}库" if cushions > 0 else "直球"
            
            print(f"    [{idx+1}] 目标={candidate['target_id']} -> {candidate['pocket_id']} "
                  f"| 类型={candidate['shot_type']}({cushion_str}) "
                  f"| V0={action['V0']:.2f} phi={action['phi']:.1f}° "
                  f"| 成功率={score_result['success_rate']:.0%} 犯规率={score_result['foul_rate']:.0%} "
                  f"| 走位={score_result['position_score']:.1f} "
                  f"| 总分={score_result['final_score']:.1f}")
        
        # 按 final_score 排序
        scored_candidates.sort(key=lambda x: -x['final_score'])
        return scored_candidates

    def _quick_simulate(self, action, balls, table, target_id, my_targets):
        """
        快速模拟（第一阶段筛选用）
        只模拟3次，快速判断是否有进球可能
        """
        success_count = 0
        foul_count = 0
        
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        
        for _ in range(3):  # 只模拟3次
            noisy_action = self._add_noise(action)
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            balls_state_before = {bid: ball.state.s for bid, ball in sim_balls.items()}
            
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            shot.cue.set_state(**noisy_action)
            pt.simulate(shot, inplace=True)
            
            result = self._analyze_shot_result(shot, balls_state_before, target_id, my_targets)
            
            if result['is_success']:
                success_count += 1
            if result['is_foul']:
                foul_count += 1
        
        return {
            'success_rate': success_count / 3,
            'foul_rate': foul_count / 3
        }

    def _quick_score(self, solution):
        """
        快速评分（基于几何特征，不进行模拟）
        用于预筛选候选方案
        """
        score = 100
        
        # 直球优于翻袋/踢球
        shot_type = solution['type']
        if shot_type == 'Direct':
            score += 30
        elif shot_type == 'Bank':
            score -= 10
        elif 'Kick' in shot_type:
            score -= 20
        
        # 切角越小越好
        cut_angle = solution.get('cut_angle', 0)
        score -= cut_angle * 0.5
        
        # 库边数越少越好
        cushions = solution.get('cushions', 0)
        score -= cushions * 15
        
        return score

    # ==================== 蒙特卡洛模拟 ====================
    
    def simulate_and_score(self, action, balls, table, target_id, my_targets):
        """
        对单个动作进行蒙特卡洛模拟，返回综合评分
        
        返回:
            dict: {
                'success_rate': 进球成功率,
                'foul_rate': 犯规率,
                'position_score': 走位平均分,
                'final_score': 综合评分
            }
        """
        success_count = 0
        foul_count = 0
        position_scores = []
        penalty_total = 0
        
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        
        for i in range(self.num_simulation):
            # 1. 添加噪声
            noisy_action = self._add_noise(action)
            
            # 2. 物理模拟
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # 关键修复：保存模拟前的状态（state.s），因为 pt.simulate 会 in-place 修改
            balls_state_before = {bid: ball.state.s for bid, ball in sim_balls.items()}
            
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            shot.cue.set_state(**noisy_action)
            pt.simulate(shot, inplace=True)
            
            # 3. 分析结果 - 使用保存的初始状态进行比较
            result = self._analyze_shot_result(shot, balls_state_before, target_id, my_targets)
            
            # 4. 统计
            if result['is_success']:
                success_count += 1
                # 走位评分（成功时才评）
                pos_score = self._evaluate_position(
                    result['cue_final_pos'],
                    my_targets,
                    shot.balls,
                    table
                )
                position_scores.append(pos_score)
            
            if result['is_foul']:
                foul_count += 1
                penalty_total += result['penalty']
        
        # 汇总
        success_rate = success_count / self.num_simulation
        foul_rate = foul_count / self.num_simulation
        avg_position = np.mean(position_scores) if position_scores else 0
        avg_penalty = penalty_total / self.num_simulation
        
        # 如果存在致命犯规风险（任何一次模拟中出现 -500 以下的惩罚），直接返回极低分
        if penalty_total / self.num_simulation <= -50:  # 平均惩罚超过50说明有严重犯规
            # 检查是否有致命犯规（-500 或 -1000 的惩罚）
            fatal_penalty = penalty_total <= -500  # 累计惩罚很高说明有多次严重犯规
            if fatal_penalty:
                final_score = -1000  # 直接判死
            else:
                # 普通犯规，正常计算但惩罚权重更高
                final_score = (
                    success_rate * 100  # 进球奖励
                    - foul_rate * 50    # 增加犯规惩罚权重
                    + avg_position * 0.3  # 走位权重
                    + avg_penalty * 0.5   # 增加惩罚权重
                )
        else:
            # 综合评分公式（正常情况）
            final_score = (
                success_rate * 100  # 进球奖励
                - foul_rate * 30    # 犯规惩罚
                + avg_position * 0.3  # 走位权重
                + avg_penalty * 0.3   # 惩罚项
            )
        
        return {
            'success_rate': success_rate,
            'foul_rate': foul_rate,
            'position_score': avg_position,
            'final_score': final_score
        }

    def _add_noise(self, action):
        """为动作添加高斯噪声"""
        return {
            'V0': np.clip(action['V0'] + np.random.normal(0, self.noise_std['V0']), 0.5, 8.0),
            'phi': (action['phi'] + np.random.normal(0, self.noise_std['phi'])) % 360,
            'theta': np.clip(action.get('theta', 0) + np.random.normal(0, self.noise_std['theta']), 0, 90),
            'a': np.clip(action.get('a', 0) + np.random.normal(0, self.noise_std['a']), -0.5, 0.5),
            'b': np.clip(action.get('b', 0) + np.random.normal(0, self.noise_std['b']), -0.5, 0.5)
        }

    def _analyze_shot_result(self, shot, balls_state_before, target_id, my_targets):
        """
        分析单次模拟结果
        
        参数:
            shot: 模拟后的 System 对象
            balls_state_before: 模拟前每个球的 state.s 字典 {bid: int}
            target_id: 目标球ID
            my_targets: 我方所有目标球
        
        返回:
            dict: {
                'is_success': 是否成功进球,
                'is_foul': 是否犯规,
                'penalty': 犯规惩罚分,
                'cue_final_pos': 白球最终位置,
                'pocketed': 进袋球列表
            }
        """
        # 进袋分析 - 使用 balls_state_before (dict of state.s)
        new_pocketed = [
            bid for bid, b in shot.balls.items() 
            if b.state.s == 4 and balls_state_before.get(bid, 0) != 4
        ]
        cue_pocketed = 'cue' in new_pocketed
        eight_pocketed = '8' in new_pocketed
        target_pocketed = target_id in new_pocketed
        
        # 事件分析
        first_contact = None
        cue_hit_cushion = False
        target_hit_cushion = False
        
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            
            # 首球接触
            if first_contact is None and 'cushion' not in et and 'pocket' not in et and 'cue' in ids:
                other_ids = [i for i in ids if i != 'cue']
                if other_ids:
                    first_contact = other_ids[0]
            
            # 碰库
            if 'cushion' in et:
                if 'cue' in ids:
                    cue_hit_cushion = True
                if first_contact and first_contact in ids:
                    target_hit_cushion = True
        
        # 判定成功
        is_success = target_pocketed and not cue_pocketed
        if is_success and eight_pocketed and target_id != '8':
            is_success = False  # 误打黑8
        
        # 检查是否还有其他自己的球（判断是否可以打黑8）
        remaining_own = [t for t in my_targets if t != '8' and balls_state_before.get(t, 0) != 4]
        can_shoot_eight = len(remaining_own) == 0
        
        # 判定犯规 - 使用极高惩罚避免致命错误
        is_foul = False
        penalty = 0
        
        # === 致命犯规 (直接判负) - 极高惩罚 ===
        
        # 1. 黑8和白球同时落袋 -> 判负
        if eight_pocketed and cue_pocketed:
            is_foul = True
            penalty = -1000  # 致命！直接判负
        
        # 2. 自身球未清空前打进黑8 -> 判负
        elif eight_pocketed and not can_shoot_eight:
            is_foul = True
            penalty = -1000  # 致命！直接判负
        
        # 3. 打黑8时白球落袋 -> 判负
        elif target_id == '8' and cue_pocketed:
            is_foul = True
            penalty = -800  # 非常严重
        
        # === 普通犯规 ===
        
        # 4. 普通白球落袋
        elif cue_pocketed:
            is_foul = True
            penalty = -200  # 严重但不致命
        
        # 5. 意外碰进黑8 (不是目标球)
        elif eight_pocketed and target_id != '8':
            is_foul = True
            penalty = -500  # 可能致命，取决于是否清空
        
        # 6. 空杆
        elif first_contact is None:
            is_foul = True
            penalty = -50
        
        # 7. 未碰库
        elif not new_pocketed and not (cue_hit_cushion or target_hit_cushion):
            is_foul = True
            penalty = -30
        
        # 白球最终位置
        cue_final_pos = shot.balls['cue'].state.rvw[0].copy()
        
        return {
            'is_success': is_success,
            'is_foul': is_foul,
            'penalty': penalty,
            'cue_final_pos': cue_final_pos,
            'pocketed': new_pocketed
        }

    # ==================== 走位评分 ====================
    
    def _evaluate_position(self, cue_pos, my_targets, balls_after, table):
        """
        评估白球停点的质量
        
        评估维度:
        1. 对剩余目标球的可打性（调用几何求解器）
        2. 贴库惩罚
        3. 中心区域奖励
        
        返回:
            float: 走位分数 (大约 0-100)
        """
        score = 50  # 基础分
        R = 0.028575
        
        # 1. 检查对每个剩余目标球的可打性
        remaining = [t for t in my_targets if t in balls_after and balls_after[t].state.s != 4]
        if not remaining:
            remaining = ['8']  # 清台后打黑8
        
        # 构造虚拟白球
        virtual_cue = VirtualBall('cue', cue_pos, R)
        
        # 计算下一杆的可行解数量
        total_solutions = 0
        for target_id in remaining[:3]:  # 只检查前3个目标（效率）
            if target_id not in balls_after:
                continue
            target_ball = balls_after[target_id]
            if target_ball.state.s == 4:
                continue
                
            for pocket_id, pocket in table.pockets.items():
                solutions = self.solve_shot_parameters(
                    virtual_cue, target_ball, pocket, balls_after, table
                )
                total_solutions += len(solutions)
        
        # 方案越多，走位越好
        score += min(total_solutions * 8, 40)  # 上限 40 分
        
        # 2. 贴库惩罚
        dist_to_rail = min(
            cue_pos[0] - R, table.w - R - cue_pos[0],
            cue_pos[1] - R, table.l - R - cue_pos[1]
        )
        if dist_to_rail < 0.03:  # 3cm 内算贴库
            score -= 25
        elif dist_to_rail < 0.08:  # 8cm 内轻微惩罚
            score -= 10
        
        # 3. 中心区域奖励（便于任意方向出杆）
        center = np.array([table.w / 2, table.l / 2, 0])
        dist_to_center = np.linalg.norm(cue_pos - center)
        max_dist = np.sqrt((table.w / 2) ** 2 + (table.l / 2) ** 2)
        if dist_to_center < max_dist * 0.3:
            score += 15
        elif dist_to_center < max_dist * 0.5:
            score += 5
        
        return score

    # ==================== 防守评估 ====================
    
    def evaluate_defense_options(self, balls, my_targets, opp_targets, table):
        """
        评估防守方案
        
        核心思想: 假设对手和我一样强，用 evaluate_attack_options 模拟对手视角
                  对手能拿的分越低 → 我的防守越成功
        """
        cue_ball = balls['cue']
        
        # 找一个合法的击打目标（必须先碰自己的球）
        legal_target_id = my_targets[0] if my_targets else None
        if legal_target_id is None or legal_target_id not in balls:
            return []
        
        legal_target = balls[legal_target_id]
        
        # 获取防守几何方案（已按启发式分数排序）
        defense_solutions = self.solve_defense_parameters(
            cue_ball, legal_target, balls, table, opp_targets
        )
        
        # 只评估前5个高分方案（减少计算量）
        top_defenses = defense_solutions[:5]
        print(f"  [防守求解] 共找到 {len(defense_solutions)} 个防守方案，评估前 {len(top_defenses)} 个")
        
        scored_defenses = []
        
        for idx, sol in enumerate(top_defenses):
            action = {
                'V0': sol.get('V0', self.default_v0),
                'phi': sol['phi'],
                'theta': 0,
                'a': 0,
                'b': 0
            }
            
            # 模拟一次防守，获取白球停点
            sim_result = self._simulate_defense_once(action, balls, table, legal_target_id)
            
            if sim_result['is_foul']:
                # 犯规的防守是失败的
                scored_defenses.append({
                    'action': action,
                    'defense_score': -100,
                    'strategy': sol.get('strategy', 'Unknown'),
                    'reason': 'Foul'
                })
                print(f"    [防守{idx+1}] {sol.get('strategy', 'Unknown')} | V0={action['V0']:.2f} phi={action['phi']:.1f}° | 犯规! 得分=-100")
                continue
            
            cue_final_pos = sim_result['cue_final_pos']
            balls_after = sim_result['balls_after']
            
            # 简化的对手评估：只统计对手有多少个可打的方案（几何上）
            # 不做完整模拟，大幅减少计算量
            opp_shot_count = 0
            for opp_id in opp_targets:
                if opp_id in balls_after and balls_after[opp_id].state.s != 4:
                    opp_target = balls_after[opp_id]
                    for pid, pkt in table.pockets.items():
                        opp_sols = self.solve_shot_parameters(balls_after['cue'], opp_target, pkt, balls_after, table)
                        # 只计算直球方案
                        direct_sols = [s for s in opp_sols if s['type'] == 'Direct' and s.get('cut_angle', 90) < 60]
                        opp_shot_count += len(direct_sols)
            
            # 防守得分：对手可打方案越少越好
            if opp_shot_count == 0:
                defense_score = 100  # 完美斯诺克
            else:
                defense_score = 60 - opp_shot_count * 5  # 每个对手方案扣5分
            
            # 加入启发式预评分
            defense_score += sol.get('score', 0) * 0.1
            
            scored_defenses.append({
                'action': action,
                'defense_score': defense_score,
                'strategy': sol.get('strategy', 'Unknown'),
                'predicted_stop': cue_final_pos,
                'opp_shot_count': opp_shot_count
            })
            
            print(f"    [防守{idx+1}] {sol.get('strategy', 'Unknown')} | V0={action['V0']:.2f} phi={action['phi']:.1f}° "
                  f"| 白球停点=({cue_final_pos[0]:.2f}, {cue_final_pos[1]:.2f}) "
                  f"| 对手直球方案={opp_shot_count} | 防守得分={defense_score:.1f}")
        
        # 按分数排序
        scored_defenses.sort(key=lambda x: -x['defense_score'])
        return scored_defenses

    def _simulate_defense_once(self, action, balls, table, legal_target_id):
        """
        模拟一次防守击球
        
        返回:
            dict: {
                'is_foul': 是否犯规,
                'cue_final_pos': 白球最终位置,
                'balls_after': 模拟后的球状态
            }
        """
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        shot.cue.set_state(**action)
        pt.simulate(shot, inplace=True)
        
        # 分析结果
        new_pocketed = [
            bid for bid, b in shot.balls.items() 
            if b.state.s == 4 and sim_balls[bid].state.s != 4
        ]
        cue_pocketed = 'cue' in new_pocketed
        
        # 检查首球接触
        first_contact = None
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if 'cushion' not in et and 'pocket' not in et and 'cue' in ids:
                other_ids = [i for i in ids if i != 'cue']
                if other_ids:
                    first_contact = other_ids[0]
                    break
        
        is_foul = cue_pocketed or first_contact is None
        
        return {
            'is_foul': is_foul,
            'cue_final_pos': shot.balls['cue'].state.rvw[0].copy(),
            'balls_after': shot.balls
        }

    @staticmethod
    def get_mirror(point, rail):
        p = point.copy()
        p[rail['axis']] = 2 * rail['val'] - p[rail['axis']]
        return p
        
    @staticmethod
    def get_cushion_path(start_pos, end_pos, rail_sequence):
        # 1. 从后往前生成镜像目标点
        mirrored_targets = []
        current_target = end_pos
        for rail in reversed(rail_sequence):
            current_target = NewAgent.get_mirror(current_target, rail)
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
            max_kick_cushions = 2
            for n in range(1, max_kick_cushions + 1):
                all_seqs = []
                for seq in itertools.product(rails, repeat=n):
                    valid = True
                    for i in range(len(seq)-1):
                        if seq[i]['name'] == seq[i+1]['name']: valid = False; break
                    if valid: all_seqs.append(seq)
                
                for seq in all_seqs:
                    path_points = NewAgent.get_cushion_path(start_pos, end_pos, seq)
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
        dist_tp = np.linalg.norm(target_to_pocket)
        
        # 保护：防止距离为0导致除零错误
        if dist_tp < 1e-6:
            return []  # 目标球已在袋口，无需击球
        
        u_tp = target_to_pocket / dist_tp
        ghost_direct = target_pos - u_tp * (2 * R)
        
        if is_segment_clear(target_pos, pocket_pos):
             target_strategies.append({
                 'type': 'Direct',
                 'ghost': ghost_direct,
                 'u_target_out': u_tp, # 目标球被击打后的期望方向
                 'seq': []
             })
             
        # A2. Target Bank
        max_bank_cushions = 2
        for n in range(1, max_bank_cushions + 1):
            all_seqs = []
            for seq in itertools.product(rails, repeat=n):
                valid = True
                for i in range(len(seq)-1):
                    if seq[i]['name'] == seq[i+1]['name']: valid = False; break
                if valid: all_seqs.append(seq)
            
            for seq in all_seqs:
                path_points = NewAgent.get_cushion_path(target_pos, pocket_pos, seq)
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
                        'cue_seq': cp['seq'],
                        'target_seq': strat['seq'],
                        'cut_angle': cut_angle,
                        'ghost': ghost,
                        'u_arrival': cp['u_arrival'],
                        'u_target_out': u_target_out
                    })

        return solutions

    def solve_runout_parameters(self, cue_ball, target_ball, next_target_ball, balls, table, opponent_targets=None):
        """
        [Member B] 进攻走位求解器 (Run-out Solver)
        
        功能：
            在打进当前球(target_ball)的基础上，寻找最佳走位，以便进攻下一颗球(next_target_ball)。
            如果无法进攻，则尝试防守 (调用 solve_defense_parameters)。
            
        参数：
            cue_ball: 白球
            target_ball: 当前目标球
            next_target_ball: 下一颗目标球 (Ball对象) 或 None
            balls: 所有球状态
            table: 球桌对象
            opponent_targets: 对手目标球ID列表 (用于防守计算)
            
        返回：
            list: 包含评分的击球方案, 按推荐程度排序
        """
        if balls is None or table is None: return []
        
        pockets = table.pockets
        candidates = []
        R = cue_ball.params.R
        
        # 1. 获取当前球的所有进球方案
        for pocket_id, pocket in pockets.items():
            shots = self.solve_shot_parameters(cue_ball, target_ball, pocket, balls, table)
            for shot in shots:
                shot['pocket_id'] = pocket_id
                candidates.append(shot)
        
        # 2. 如果没有进球方案，转入防守模式
        if not candidates:
            # print("[Run-out] 无进攻机会，切换至防守模式...")
            defense_solutions = self.solve_defense_parameters(cue_ball, target_ball, balls, table, opponent_targets)
            # 统一格式返回
            return defense_solutions
            
        # 3. 评估走位 (Run-out Logic)
        scored_candidates = []
        
        for shot in candidates:
            # --- 基础分：进球难度 ---
            # 直球最稳，Kick/Bank 风险大
            base_score = 100
            if shot['type'] == 'Bank': base_score -= 30
            elif 'Kick' in shot['type']: base_score -= 40
            
            # 切角过大容易失误
            if shot['cut_angle'] > 60: base_score -= (shot['cut_angle'] - 60)
            
            # --- 走位分：为下一杆做准备 ---
            position_score = 0
            predicted_stop_pos = None
            
            if next_target_ball:
                # 计算切线方向 (Tangent Line) - 假设定杆(Stun)
                # u_out = u_arrival - (u_arrival . u_target_out) * u_target_out
                u_arr = shot['u_arrival']
                u_tgt = shot['u_target_out']
                dot = np.dot(u_arr, u_tgt)
                u_out_vec = u_arr - dot * u_tgt
                
                # 归一化分离方向
                norm_out = np.linalg.norm(u_out_vec)
                if norm_out < 1e-6:
                    u_out = np.zeros(3) # 正面撞击，停在原地
                else:
                    u_out = u_out_vec / norm_out
                
                # 寻找切线上最佳停点
                # 假设白球能在切线方向滚动 0.2m ~ 1.5m
                # 我们采样几个点，看哪个点对 next_target 最好
                best_spot_score = -float('inf')
                best_spot = None
                
                ghost_pos = shot['ghost']
                
                # 采样距离: 0 (Stop), 0.3, 0.6, 0.9, 1.2 (Follow/Draw adjusted tangent)
                # 注意：纯定杆只能沿切线。推杆/拉杆会偏离切线。
                # 这里简化模型：只考虑切线上的点 (对于大角度切球，切线是主要分量)
                
                for dist in [0.0, 0.3, 0.6, 0.9, 1.2]:
                    stop_pos = ghost_pos + u_out * dist
                    
                    # 检查是否出界
                    if not (R < stop_pos[0] < table.w - R and R < stop_pos[1] < table.l - R):
                        continue
                    
                    # [NEW] 模拟下一杆，计算可行解数量
                    next_shot_count = 0
                    if next_target_ball:
                        sim_balls = balls.copy()
                        sim_balls['cue'] = VirtualBall('cue', stop_pos, R)
                        # 移除已打进的目标球
                        if target_ball.id in sim_balls:
                            del sim_balls[target_ball.id]
                        
                        if next_target_ball.id in sim_balls:
                            sim_next_target = sim_balls[next_target_ball.id]
                            for pid, pkt in pockets.items():
                                 # 调用求解器计算下一杆的可行解
                                 next_sols = self.solve_shot_parameters(sim_balls['cue'], sim_next_target, pkt, sim_balls, table)
                                 next_shot_count += len(next_sols)
                        
                    # [NEW] 基于解数量的评分
                    # 解越多，走位越好。
                    simulation_score = next_shot_count * 5
                    
                    # A. 是否可见 (Clear Path) - 保留作为快速过滤
                    next_pos = next_target_ball.state.rvw[0]
                    
                    # 关键修正：检查下一杆时，当前目标球(target_ball)已经被打进，所以应排除
                    # 同时也要排除下一颗球(next_target_ball)自己，因为它在路径终点，会被视为障碍
                    if not self.is_segment_clear_static(stop_pos, next_pos, balls, exclude_ids=[cue_ball.id, target_ball.id, next_target_ball.id]):
                        spot_score = -100 # 被阻挡
                    else:
                        spot_score = simulation_score # 使用模拟分作为基础
                        
                        # B. 距离适中 (太远难打，太近不好运杆)
                        d_next = np.linalg.norm(next_pos - stop_pos)
                        if d_next < 0.3: spot_score -= 20 # 太近
                        elif d_next > 1.5: spot_score -= (d_next - 1.5) * 10 # 太远
                        else: spot_score += 20 # 舒适区
                        
                        # C. 进球角度 (Next Cut Angle) - 保留作为质量补充
                        best_pocket_angle_score = -50
                        for pid, pkt in pockets.items():
                            # 检查 Next -> Pocket 是否通畅 (同样排除当前目标球)
                            if self.is_segment_clear_static(next_pos, pkt.center, balls, exclude_ids=[cue_ball.id, next_target_ball.id, target_ball.id]):
                                # 计算 Next Cut Angle
                                vec_np = pkt.center - next_pos
                                vec_np[2] = 0
                                u_np = vec_np / np.linalg.norm(vec_np)
                                
                                vec_cn = next_pos - stop_pos
                                vec_cn[2] = 0
                                dist_cn = np.linalg.norm(vec_cn)
                                if dist_cn > 1e-6:
                                    u_cn = vec_cn / dist_cn
                                    # Cut Angle
                                    angle = np.degrees(np.arccos(np.clip(np.dot(u_cn, u_np), -1, 1)))
                                    
                                    # 理想角度：15~45度 (方便再次走位)，直球也可以但走位稍难
                                    # 0~60度是可接受范围
                                    if angle > 70: s = -50
                                    elif angle > 50: s = 10
                                    elif angle < 10: s = 30
                                    else: s = 50 # 10-50度 最佳
                                    
                                    if s > best_pocket_angle_score:
                                        best_pocket_angle_score = s
                        
                        spot_score += best_pocket_angle_score
                    
                    if spot_score > best_spot_score:
                        best_spot_score = spot_score
                        best_spot = stop_pos
                
                if best_spot is not None:
                    position_score = best_spot_score
                    predicted_stop_pos = best_spot
                else:
                    position_score = -50 # 无法停在好位置
            else:
                # 如果没有下一颗球，只求打进，且白球不洗袋
                position_score = 0
                predicted_stop_pos = shot['ghost'] # 粗略
            
            # 总分
            shot['score'] = base_score + position_score
            shot['predicted_stop_pos'] = predicted_stop_pos
            scored_candidates.append(shot)
            
        # 排序
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        return scored_candidates

    def trace_ray_with_rebound(self, start_pos, direction_u, distance, table_w, table_l, R):
        """
        Simulate a rolling ball path with wall rebounds.
        Returns:
            final_pos (np.array): The stop position.
            path_points (list of np.array): The list of vertices (start, bounce1, bounce2, ..., end).
        """
        pos = start_pos.copy()
        u = direction_u.copy()
        remaining_dist = distance
        path = [pos.copy()]
        
        # Max rebounds to prevent infinite loops (though distance limits it naturally)
        for _ in range(5):
            if remaining_dist <= 1e-4: break
            
            # 1. Find collision time to each wall
            # Walls: x=R, x=W-R, y=R, y=L-R
            # To handle cushion compression or physics radius, we use R.
            
            t_min = float('inf')
            wall_idx = -1 # 0:Left, 1:Right, 2:Bottom, 3:Top
            
            # Left (x=R)
            if u[0] < -1e-6:
                t = (R - pos[0]) / u[0]
                if 0 <= t < t_min: t_min = t; wall_idx = 0
            # Right (x=W-R)
            elif u[0] > 1e-6:
                t = (table_w - R - pos[0]) / u[0]
                if 0 <= t < t_min: t_min = t; wall_idx = 1
            
            # Bottom (y=R)
            if u[1] < -1e-6:
                t = (R - pos[1]) / u[1]
                if 0 <= t < t_min: t_min = t; wall_idx = 2
            # Top (y=L-R)
            elif u[1] > 1e-6:
                t = (table_l - R - pos[1]) / u[1]
                if 0 <= t < t_min: t_min = t; wall_idx = 3
                
            # Check if we stop before hitting wall
            if remaining_dist <= t_min:
                pos = pos + u * remaining_dist
                path.append(pos.copy())
                break
            else:
                # Hit wall
                pos = pos + u * t_min
                path.append(pos.copy())
                remaining_dist -= t_min
                
                # Reflect
                if wall_idx == 0 or wall_idx == 1: # Left/Right -> Reflect X
                    u[0] = -u[0]
                elif wall_idx == 2 or wall_idx == 3: # Top/Bottom -> Reflect Y
                    u[1] = -u[1]
                
                # Energy loss on cushion? (Optional)
                # remaining_dist *= 0.8
        
        return pos, path

    def solve_defense_parameters(self, cue_ball, target_ball, balls, table, opponent_targets=None):
        """
        [Member B 实现] 防守求解器
        
        功能：
            寻找防守击球方案，使白球停在安全位置（制造障碍或远台）。
            
        参数：
            cue_ball: 白球对象
            target_ball: 我方必须击打的球（合法目标）
            balls: 所有球状态
            table: 球桌对象
            opponent_targets: 对手目标球ID列表 (用于评估Snooker效果)
            
        返回：
            list: [{'phi': float, 'V0': float, 'type': 'Safety', 'score': float, 'strategy': str}]
        """
        if balls is None or table is None: return []
        
        solutions = []
        R = cue_ball.params.R
        cue_pos = cue_ball.state.rvw[0]
        target_pos = target_ball.state.rvw[0]
        
        # 1. 尝试 Direct Hit 防守
        direct_possible = self.is_path_clear(cue_ball, target_ball, None, balls)
        
        if direct_possible:
            vec_ct = target_pos - cue_pos
            vec_ct[2] = 0
            dist_ct = np.linalg.norm(vec_ct)
            u_ct = vec_ct / dist_ct
            angle_ct = np.degrees(np.arctan2(u_ct[1], u_ct[0]))
            
            # 搜索：不同的切角 和 不同的力度
            # 切角
            for cut_angle in range(-80, 81, 5): # 略微减小范围，避免85度过于极限
                check_angle = angle_ct + 180 + cut_angle 
                ghost_u = np.array([np.cos(np.radians(check_angle)), np.sin(np.radians(check_angle)), 0])
                ghost_pos = target_pos + ghost_u * (2 * R)
                
                # Check if Ghost Position is physically valid (on table)
                if not (R - 1e-4 <= ghost_pos[0] <= table.w - R + 1e-4 and 
                        R - 1e-4 <= ghost_pos[1] <= table.l - R + 1e-4):
                    continue

                # 检查白球能否直达该 Ghost
                if not self.is_segment_clear_static(cue_pos, ghost_pos, balls, exclude_ids=[cue_ball.id, target_ball.id]):
                    continue
                
                # 计算出射方向
                vec_cg = ghost_pos - cue_pos
                vec_cg[2] = 0
                dist_cg = np.linalg.norm(vec_cg)
                u_cg = vec_cg / dist_cg
                phi = np.degrees(np.arctan2(u_cg[1], u_cg[0])) % 360
                
                u_n = -ghost_u
                dot = np.dot(u_cg, u_n)
                u_out_vec = u_cg - dot * u_n
                norm_out = np.linalg.norm(u_out_vec)
                u_out = u_out_vec / norm_out if norm_out > 1e-6 else np.zeros(3)
                
                # 力度搜索 (对应滚动距离)
                # 假设我们可以控制白球撞击后滚动 0.5m, 1.0m, 1.5m, 2.0m, 2.5m, 3.0m
                for roll_dist in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                    pred_stop_pos, roll_path = self.trace_ray_with_rebound(ghost_pos, u_out, roll_dist, table.w, table.l, R)
                    
                    # [NEW] Check if roll path collides with any ball (Trajectory Validation)
                    path_collision = False
                    for i in range(len(roll_path) - 1):
                        # Exclude cue (moving) and target (hit and moving away)
                        if not self.is_segment_clear_static(roll_path[i], roll_path[i+1], balls, exclude_ids=[cue_ball.id, target_ball.id]):
                            path_collision = True; break
                    if path_collision: continue

                    # 检查最终停点是否与其他球重叠 (Invalid position)
                    valid_stop = True
                    for bid, b in balls.items():
                        if bid == 'cue': continue # Cue is moving
                        if np.linalg.norm(pred_stop_pos - b.state.rvw[0]) < 2*R:
                            valid_stop = False; break
                    if not valid_stop: continue

                    # Score
                    score = 0
                    if opponent_targets:
                        min_dist = float('inf')
                        for opp_id in opponent_targets:
                            if opp_id in balls:
                                d = np.linalg.norm(pred_stop_pos - balls[opp_id].state.rvw[0])
                                if d < min_dist: min_dist = d
                        score += min_dist * 10
                        
                        snookered = 0
                        for opp_id in opponent_targets:
                            if opp_id in balls:
                                # 检查从 停点 到 对手目标球 是否被挡
                                # 注意：要排除自己(因为是停点)和该对手球
                                # 关键：检查谁在挡？可能是 Target Ball 刚被我们打跑了？
                                # 这是一个动态问题。简化假设：Target Ball 被打远了，或者就在原地附近(如果是轻打)。
                                # 如果是薄切，Target Ball 动得不多。
                                # 这里我们还是用静态 balls 检查，但排除 target_ball (假设它不在阻挡线上，或者它就是阻挡者？)
                                # 更好的做法：假设 Target Ball 移动了一段距离。
                                # 简化：排除 target_ball，只看其他障碍物。
                                if not self.is_segment_clear_static(pred_stop_pos, balls[opp_id].state.rvw[0], balls, exclude_ids=[opp_id, target_ball.id]):
                                    snookered += 1
                        score += snookered * 50
                    
                    # 贴库奖励
                    dist_to_rail = min(pred_stop_pos[0]-R, table.w-R-pred_stop_pos[0], pred_stop_pos[1]-R, table.l-R-pred_stop_pos[1])
                    if dist_to_rail < 1.5 * R: score += 30
                    
                    if score > 20:
                        # V0 estimate: v^2 = 2*a*d. a approx 0.5? let's say V0 = sqrt(roll_dist * 2) + impact loss
                        # Simplified V0 map
                        v0_est = 1.0 + roll_dist * 0.8
                        
                        solutions.append({
                            'type': 'Safety',
                            'phi': phi,
                            'V0': v0_est,
                            'score': score,
                            'strategy': 'Direct-Safety',
                            'cut_angle': abs(cut_angle),
                            'ghost': ghost_pos,
                            'predicted_stop_pos': pred_stop_pos,
                            'roll_dist': roll_dist,
                            'roll_path': roll_path
                        })

        # 2. 增强版 Kick Safety (支持 1-2 库)
        max_safety_cushions = 2
        
        import itertools 
        
        # Define rails locally
        rails = [
            {'name': 'left',   'val': 0,       'axis': 0, 'limit': (0, table.l)},
            {'name': 'right',  'val': table.w, 'axis': 0, 'limit': (0, table.l)},
            {'name': 'bottom', 'val': 0,       'axis': 1, 'limit': (0, table.w)},
            {'name': 'top',    'val': table.l, 'axis': 1, 'limit': (0, table.w)}
        ]
        
        for n in range(1, max_safety_cushions + 1):
            all_seqs = []
            for seq in itertools.product(rails, repeat=n):
                valid = True
                for i in range(len(seq)-1):
                    if seq[i]['name'] == seq[i+1]['name']: valid = False; break
                if valid: all_seqs.append(seq)
            
            for seq in all_seqs:
                # Target is just a point we want to HIT.
                # Use get_cushion_path
                path_points = self.get_cushion_path(cue_pos, target_pos, seq)
                if path_points:
                    # Check clearance
                    path_clear = True
                    for i in range(len(path_points)-1):
                        p1, p2 = path_points[i], path_points[i+1]
                        # For safety, we just want to reach target ball.
                        # Exclude target from obstacles (it's the goal).
                        # Exclude cue from obstacles.
                        # [Modified] Add safety margin to avoid visual overlap or physics edge cases
                        if not self.is_segment_clear_static(p1, p2, balls, exclude_ids=[cue_ball.id, target_ball.id], margin=0.015):
                            path_clear = False; break
                    
                    if path_clear:
                        # Found a path!
                        p1 = path_points[1]
                        vec_shot = p1 - cue_pos
                        phi = np.degrees(np.arctan2(vec_shot[1], vec_shot[0])) % 360
                        
                        # Score
                        score = 40 + n * 10 # More cushions = harder but cooler/more unexpected
                        
                        # Snooker bonus?
                        # Assume stop at target (rough estimate)
                        # Kick Safety 通常很难控制停点，但如果我们要给用户信心，
                        # 我们假设它停在 Target 附近 (撞击后动能损失大)
                        pred_stop_pos = target_pos
                        
                        # [NEW] 模拟评估 (Kick Safety)
                        # ... (Same logic as before) ...
                        opp_virtual_cue = VirtualBall('cue', pred_stop_pos, R)
                        opp_shot_count = 0
                        if opponent_targets:
                            sim_balls = balls.copy()
                            sim_balls['cue'] = opp_virtual_cue
                            
                            for opp_target_id in opponent_targets:
                                if opp_target_id in sim_balls:
                                    opp_target = sim_balls[opp_target_id]
                                    for pid, pkt in table.pockets.items():
                                        opp_sols = self.solve_shot_parameters(opp_virtual_cue, opp_target, pkt, sim_balls, table)
                                        opp_shot_count += len(opp_sols)
                        
                        # 评分：对手解越少，分数越高
                        if opp_shot_count == 0:
                            score += 200 # Absolute Snooker
                        else:
                            score += max(0, 100 - opp_shot_count * 5)
                            
                        # 原有的 heuristic 也可以保留作为补充
                        if opponent_targets:
                            snookered = 0
                            for opp_id in opponent_targets:
                                if opp_id in balls:
                                    if not self.is_segment_clear_static(pred_stop_pos, balls[opp_id].state.rvw[0], balls, exclude_ids=[opp_id]):
                                        snookered += 1
                            score += snookered * 20
                            
                        solutions.append({
                            'type': 'Safety',
                            'phi': phi,
                            'V0': 3.5, # Multi-rail needs more power
                            'score': score,
                            'strategy': f'{n}-Rail-Kick-Safety',
                            'cut_angle': 0,
                            'ghost': target_pos, # Approx
                            'predicted_stop_pos': pred_stop_pos,
                            'cue_seq': [r['name'] for r in seq]
                        })
        
        # 按分数排序
        solutions.sort(key=lambda x: x['score'], reverse=True)
        return solutions # 返回所有解，由调用者决定取多少

    def is_segment_clear_static(self, p1, p2, balls, exclude_ids=[], margin=0.0):
        """静态方法的路径检测，方便内部调用"""
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-6: return True
        u = vec / length
        R = 0.028575 # Hardcoded or passed
        
        threshold = 2 * R + margin

        for bid, ball in balls.items():
            if ball.state.s == 4: continue
            if bid in exclude_ids: continue
            
            pos = ball.state.rvw[0]
            ap = pos - p1
            proj = np.dot(ap, u)
            if proj < 0 or proj > length:
                dist = min(np.linalg.norm(pos - p1), np.linalg.norm(pos - p2))
            else:
                dist = np.linalg.norm(ap - proj * u)
            
            if dist < threshold:
                return False
        return True

    def is_path_clear(self, cue_ball, target_ball, pocket, balls):
        return self.is_segment_clear_static(cue_ball.state.rvw[0], target_ball.state.rvw[0], balls, [cue_ball.id, target_ball.id])
