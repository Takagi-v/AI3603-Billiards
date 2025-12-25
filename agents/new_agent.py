import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime
import copy
import signal

import cma  # CMA-ES 优化库

from .agent import Agent

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

# ============ 走位评分工具函数 (从 agent_new.py 移植) ============

def get_ball_position(ball) -> np.ndarray:
    """提取球的位置坐标 (x, y, z)"""
    return ball.state.rvw[0].copy()


def get_pocket_position(pocket, table=None) -> np.ndarray:
    """
    提取袋口的位置坐标（带中袋偏移修正）
    
    中袋（side pockets）的物理袋口边缘比几何中心更靠外，
    为了提高瞄准稳定性，对中袋的瞄准点向球桌中心方向偏移。
    
    参数:
        pocket: 袋口对象
        table: 球桌对象（可选，用于判断球桌中心位置）
    
    返回:
        np.ndarray: 袋口位置 (x, y, z)
    """
    # 获取原始袋口位置
    if hasattr(pocket, 'center'):
        pos = np.array([pocket.center[0], pocket.center[1], 0])
    elif hasattr(pocket, 'a'):
        pos = np.array([pocket.a, pocket.b, 0])
    else:
        pos = np.array([pocket.x, pocket.y, 0])
    
    # 对中袋（side pockets）添加偏移
    # 中袋的ID通常包含 'c' (center)，如 'lc', 'rc'
    pocket_id = getattr(pocket, 'id', '')
    if 'c' in pocket_id.lower():
        # 中袋偏移：向球桌中心方向偏移
        # 袋口半径约 0.0645m，偏移半径的 50%
        r_side = 0.0645
        offset = r_side * 0.5
        
        # 判断是左中袋还是右中袋
        if pos[0] < 0.5:  # 左中袋
            pos[0] += offset  # 向右（中心）偏移
        else:  # 右中袋
            pos[0] -= offset  # 向左（中心）偏移
    
    return pos


def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个向量之间的夹角（度数）
    
    返回:
        float: 夹角，范围 [0, 180]
    """
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-9 or v2_norm < 1e-9:
        return 0.0
    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def point_to_segment_distance(point, seg_start, seg_end):
    """计算点到线段的距离"""
    point = np.array(point)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)
    
    v = seg_end - seg_start
    w = point - seg_start
    
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(point - seg_start)
    
    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(point - seg_end)
    
    b = c1 / c2
    pb = seg_start + b * v
    return np.linalg.norm(point - pb)

# ============================================


class NewAgent(Agent):
    """
    重构后的 NewAgent - 模块化决策架构
    
    架构:
    1. solve_shot_parameters - 纯几何求解
    2. simulate_and_score - 蒙特卡洛模拟+评分
    3. _evaluate_position - 走位评分
    4. evaluate_attack_options - 评估所有进攻选项
    5. _bayesian_refine - 贝叶斯小范围微调
    6. decision - 精简的主入口
    """
    
    def __init__(self):
        super().__init__()
        # 蒙特卡洛模拟参数
        self.num_simulation = 10       # 精细模拟次数
        
        # 噪声参数（与环境一致）
        self.noise_std = {
            'V0': 0.1, 'phi': 0.1, 'theta': 0.1, 'a': 0.003, 'b': 0.003
        }
        
        # 决策参数
        self.base_attack_threshold = 55  # 进攻门槛
        self.default_v0 = 2.4  # 默认力度
        
        # ==================== 4档力度系统 ====================
        # 4档力度：小力、中力、大力、极大力（删除过小力度档位）
        self.power_levels = {
            'soft': 2.5,        # 小力：近距离进球
            'medium': 4.0,      # 中力：中距离进球
            'hard': 5.5,        # 大力：远距离进球
            'very_hard': 7.0,   # 极大力：超远距离/穿透
        }
        self.power_names = ['soft', 'medium', 'hard', 'very_hard']
        
        # ==================== CMA-ES 微调配置 ====================
        self.CMA_ES_ENABLE = True    # 是否启用 CMA-ES 微调
        self.CMA_ES_MAXITER = 3      # CMA-ES 最大迭代次数
        self.CMA_ES_POPSIZE = 6      # CMA-ES 种群大小
        
        print("[NewAgent] 架构初始化完成（4档力度 + Top-15精细模拟 + CMA-ES微调 + 致命检查）")

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
            action = self._break_shot_action(balls, table, my_targets)
            print(f"\n  >>> 决策: 故意犯规让对手开球!")
            print(f"      力度: V0={action['V0']:.1f}, 角度: phi={action['phi']:.1f}°")
            print(f"{'='*60}\n")
            return action
        
        # 2. 评估所有进攻选项
        print(f"\n  === 进攻评估 ===")
        attack_options = self.evaluate_attack_options(balls, legal_targets, table)
        
        # 3. 动态阈值
        remaining_own = len(legal_targets)
        attack_threshold = self._get_attack_threshold(remaining_own)
        
        # 筛选符合条件的进攻方案：成功率 >= 60% 斩杀线
        MIN_SUCCESS_RATE = 0.6  # 成功率斩杀线
        valid_attacks = [opt for opt in attack_options if opt['success_rate'] >= MIN_SUCCESS_RATE]
        
        best_attack_score = valid_attacks[0]['final_score'] if valid_attacks else -100
        best_attack = valid_attacks[0] if valid_attacks else None
        
        if valid_attacks:
            print(f"\n  符合成功率要求(>={MIN_SUCCESS_RATE:.0%})的方案: {len(valid_attacks)}个")
            print(f"  最高分: {best_attack_score:.1f}, 阈值: {attack_threshold:.1f}")
        else:
            print(f"\n  没有成功率>={MIN_SUCCESS_RATE:.0%}的方案，切换防守")
        
        # 4. 决策
        # 4. 决策：选取最佳方案并进行微调优化
        # 即使是好方案，也通过 CMA-ES 在小范围内寻找更好的力度和杆法，优化走位
        selected_candidate = None
        is_fallback = False
        
        if best_attack and best_attack_score >= attack_threshold:
            print(f"\n  >>> 选中最佳方案 (分值: {best_attack_score:.1f})，准备微调...")
            selected_candidate = best_attack
        elif valid_attacks:
            # 也就凑合的方案
            print(f"\n  >>> 选中备选方案 (分值: {valid_attacks[0]['final_score']:.1f})，准备微调...")
            selected_candidate = valid_attacks[0]
        elif attack_options:
            # 分数很低的方案，作为 fallback
            print(f"\n  >>> 只有低分方案 (分值: {attack_options[0]['final_score']:.1f})，强行尝试微调...")
            selected_candidate = attack_options[0]
            is_fallback = True
        
        if selected_candidate:
            # === 核心改进：全时 CMA-ES 微调 ===
            # 不要直接相信几何解的固定力度，去搜索更细腻的参数
            geo_action = selected_candidate['action']
            target_id = selected_candidate['target_id']
            
            final_action = geo_action
            
            if self.CMA_ES_ENABLE:
                print(f"  [CMA-ES] 正在优化参数 (V0, phi, theta, a, b)...")
                # 增加微调的迭代次数以获得更精细的手感
                refined_action, theoretical_score = self._cma_es_refine(
                    geo_action, balls, legal_targets, table, target_id
                )
                
                # === 关键修正：鲁棒性验证 (Reality Check) ===
                # CMA-ES 是在无噪声环境下优化的，必须在带噪声环境下验证其稳定性
                # 防止出现“理论无敌，实战拉胯”的过拟合现象
                print(f"  [CMA-ES] 理论得分: {theoretical_score:.1f} -> 正在进行鲁棒性验证(Sim=5)...")
                
                verify_result = self.simulate_and_score(
                    refined_action, balls, table, target_id, legal_targets,
                    num_simulations=5, add_noise=True
                )
                verified_score = verify_result['final_score']
                verified_foul_rate = verify_result['foul_rate']
                
                print(f"  [验证结果] 实战得分: {verified_score:.1f} (犯规率: {verified_foul_rate:.0%})")
                
                # 决策逻辑：只有当实战得分更高，且犯规率可控时才采用
                # 原始方案的得分
                original_score = selected_candidate['final_score']
                
                # 接受条件：
                # 1. 验证分数显著高于原方案
                # 2. 或者验证分数持平，但理论分数很高（说明潜力大且没变坏） - 慎重，优先看实战
                # 3. 必须：犯规率不能过高 (>20% 直接枪毙)
                
                if verified_foul_rate > 0.2:
                    print(f"  [CMA-ES] ❌ 拒绝：实战犯规率过高 ({verified_foul_rate:.0%})，由于噪声导致的不稳定")
                elif verified_score > original_score:
                    print(f"  [CMA-ES] ✅ 接受：实战效果提升 ({original_score:.1f} -> {verified_score:.1f})")
                    final_action = refined_action
                    
                    # 打印优化对比
                    print(f"      V0: {geo_action['V0']:.2f} -> {final_action['V0']:.2f}")
                    print(f"      phi: {geo_action['phi']:.2f} -> {final_action['phi']:.2f}")
                    if abs(final_action['a']) > 0.01 or abs(final_action['b']) > 0.01:
                        print(f"      Spin: (0,0) -> ({final_action['a']:.3f}, {final_action['b']:.3f})")
                else:
                    print(f"  [CMA-ES] ❌ 拒绝：实战效果未明显提升 (原方案: {original_score:.1f} vs 微调: {verified_score:.1f})")

            
            # 最终输出
            cushions = selected_candidate.get('cushions', 0)
            cushion_str = f"{cushions}库" if cushions > 0 else "直球"
            print(f"\n  >>> 最终决策: [{'强行' if is_fallback else '自信'}进攻]")
            print(f"      目标: {selected_candidate['target_id']} -> {selected_candidate['pocket_id']}")
            print(f"      类型: {selected_candidate['shot_type']} ({cushion_str})")
            print(f"      参数: V0={final_action['V0']:.2f}, phi={final_action['phi']:.2f}°, a={final_action.get('a',0):.2f}, b={final_action.get('b',0):.2f}")
            print(f"{'='*60}\n")
            return final_action

        # 5. 彻底绝望：保守防守
        print(f"\n  >>> 决策: 无任何可行进攻路线，执行保守防守")
        action = self._conservative_shot(balls, legal_targets[0], table)
        print(f"      小力轻贴目标球: V0={action['V0']:.1f}, phi={action['phi']:.1f}°")
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

    def _check_fatal_failure(self, action, balls, my_targets, table, num_checks=5):
        """
        致命失误快速检查
        
        检测方案是否存在致命风险：
        1. 白球 + 黑8 同时落袋 -> 直接判负
        2. 未清台时打进黑8 -> 直接判负
        3. 打黑8时白球落袋 -> 判负
        
        参数:
            action: 待检查的动作
            balls: 球状态
            my_targets: 己方目标球列表
            table: 球桌
            num_checks: 检查次数（带噪声模拟）
        
        返回:
            dict: {
                'is_fatal': bool - 是否存在致命风险,
                'fatal_rate': float - 致命失误概率,
                'fatal_type': str - 致命失误类型
            }
        """
        targeting_eight = (my_targets == ['8'])
        fatal_count = 0
        fatal_types = []
        
        for check_i in range(num_checks):
            try:
                # 深拷贝并模拟
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                # 第一次无噪声，后续加噪声
                if check_i > 0:
                    noisy_action = self._add_noise(action)
                else:
                    noisy_action = action
                
                shot.cue.set_state(**noisy_action)
                
                if not simulate_with_timeout(shot, timeout=2):
                    continue  # 模拟超时，跳过
                
                # 分析落袋情况
                cue_pocketed = shot.balls['cue'].state.s == 4
                eight_pocketed = shot.balls['8'].state.s == 4
                
                # 检测致命情况
                is_fatal = False
                fatal_type = None
                
                # 1. 白球 + 黑8 同时落袋
                if cue_pocketed and eight_pocketed:
                    is_fatal = True
                    fatal_type = "cue_and_8_pocketed"
                
                # 2. 未清台时打进黑8
                elif eight_pocketed and not targeting_eight:
                    is_fatal = True
                    fatal_type = "illegal_8_pocketed"
                
                # 3. 打黑8时白球落袋
                elif targeting_eight and cue_pocketed:
                    is_fatal = True
                    fatal_type = "cue_pocketed_on_8"
                
                if is_fatal:
                    fatal_count += 1
                    fatal_types.append(fatal_type)
                    
                    # 提前终止：如果致命率已经很高
                    if check_i >= 2 and fatal_count / (check_i + 1) > 0.3:
                        break
                        
            except Exception:
                continue
        
        fatal_rate = fatal_count / max(1, num_checks)
        most_common_type = max(set(fatal_types), key=fatal_types.count) if fatal_types else None
        
        return {
            'is_fatal': fatal_count > 0,
            'fatal_rate': fatal_rate,
            'fatal_type': most_common_type
        }


    def _conservative_shot(self, balls, target_id, table):
        """
        保守击球：轻打向己方目标球
        当没有找到任何可行进攻方案时使用
        
        改进：遍历己方所有球，找到路径清晰的目标，确保首球碰撞是己方球
        """
        cue_pos = balls['cue'].state.rvw[0]
        R = balls['cue'].params.R
        
        # 确定己方目标球
        if target_id in [str(i) for i in range(1, 8)]:
            my_ball_ids = [str(i) for i in range(1, 8)]
        elif target_id in [str(i) for i in range(9, 16)]:
            my_ball_ids = [str(i) for i in range(9, 16)]
        else:
            my_ball_ids = [target_id]
        
        # 遍历己方所有球，找到路径最清晰的目标
        best_target = None
        best_distance = float('inf')
        
        for bid in my_ball_ids:
            if bid not in balls or balls[bid].state.s == 4:
                continue
            
            target_pos = balls[bid].state.rvw[0]
            
            # 检查路径是否清晰（没有其他球在路径上）
            path_clear = True
            for other_bid, other_ball in balls.items():
                if other_bid in ['cue', bid] or other_ball.state.s == 4:
                    continue
                other_pos = other_ball.state.rvw[0][:2]
                
                # 检测点到线段的距离
                dist = point_to_segment_distance(other_pos, cue_pos[:2], target_pos[:2])
                if dist < 2 * R + 0.01:
                    path_clear = False
                    break
            
            if path_clear:
                # 路径清晰，计算距离
                distance = np.linalg.norm(target_pos - cue_pos)
                if distance < best_distance:
                    best_distance = distance
                    best_target = bid
        
        # 如果没有路径清晰的球，使用原始目标（可能会犯规，但至少尝试）
        if best_target is None:
            best_target = target_id
            print(f"  [保守击球] 警告：没有路径清晰的目标球，使用 {target_id}")
        else:
            print(f"  [保守击球] 选择路径清晰的目标球: {best_target}")
        
        target_pos = balls[best_target].state.rvw[0]
        
        # 计算击球角度
        direction = target_pos - cue_pos
        phi = np.degrees(np.arctan2(direction[1], direction[0])) % 360
        
        # 使用力度系统的小力档位（保守击球）
        return {
            'V0': self.power_levels['soft'],  # 小力
            'phi': phi,
            'theta': 0,
            'a': 0,
            'b': 0
        }

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

    def _break_shot_action(self, balls, table, my_targets):
        """
        开球策略：故意犯规让对手开球
        
        策略逻辑：
        - 开球犯规（白球落袋）没有任何惩罚
        - 让对手帮我们把球堆打散
        - 直接把白球打进最近的底袋
        
        实现：
        1. 找到离白球最近的底袋（corner pocket）
        2. 计算直接打入底袋的角度
        3. 使用中等力度确保落袋
        """
        cue_pos = balls['cue'].state.rvw[0]
        
        print(f"  [开球] 策略: 故意犯规让对手开球（白球落袋无惩罚）")
        
        # 找到所有底袋（corner pockets）
        corner_pockets = []
        for pocket_id, pocket in table.pockets.items():
            pocket_pos = np.array(pocket.center[:2])
            # 底袋通常在四个角落
            # 根据 pocket_id 判断或者根据位置判断
            is_corner = 'c' not in pocket_id.lower()  # 非中袋就是底袋
            if is_corner:
                dist = np.linalg.norm(pocket_pos - cue_pos[:2])
                corner_pockets.append((pocket_id, pocket_pos, dist))
        
        if not corner_pockets:
            # 如果没找到底袋，使用任意袋口
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = np.array(pocket.center[:2])
                dist = np.linalg.norm(pocket_pos - cue_pos[:2])
                corner_pockets.append((pocket_id, pocket_pos, dist))
        
        # 按距离排序，选择最近的底袋
        corner_pockets.sort(key=lambda x: x[2])
        target_pocket_id, target_pocket_pos, dist_to_pocket = corner_pockets[0]
        
        # 计算打入底袋的角度
        direction = target_pocket_pos - cue_pos[:2]
        phi = np.degrees(np.arctan2(direction[1], direction[0])) % 360
        
        # 根据距离选择力度：距离越远力度越大
        # 确保白球能够进袋
        if dist_to_pocket < 0.5:
            v0 = 3.0  # 近距离小力
        elif dist_to_pocket < 1.0:
            v0 = 4.5  # 中距离中力
        else:
            v0 = 6.0  # 远距离大力
        
        action = {
            'V0': v0,
            'phi': phi,
            'theta': 0,
            'a': 0,
            'b': 0
        }
        
        print(f"  [开球] 目标: 白球 -> {target_pocket_id} (距离={dist_to_pocket:.2f}m)")
        print(f"  [开球] 动作: V0={v0:.1f}, phi={phi:.1f}°")
        print(f"  [开球] 效果: 对手获得球权并帮我们开球散堆")
        
        return action

    # ==================== 进攻评估 ====================
    
    def evaluate_attack_options(self, balls, legal_targets, table):
        """
        评估所有进攻选项
        
        流程:
        1. 遍历所有 (目标球, 袋口, 力度档位) 组合
        2. 对每个方案进行无噪声模拟验证一次
        3. 根据模拟结果评分，筛选有效方案 Top-8
        4. 对 Top-8 进行 10 次精细模拟
        5. 返回按 final_score 排序的结果
        
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
                    # [优化] 根据几何特征提前剪枝
                    # 1. 如果切角过大 (>75度) 且不是翻袋/踢球，直接丢弃 (成功率极低)
                    if sol['type'] == 'Direct' and sol.get('cut_angle', 0) > 75:
                        continue
                    
                    # 根据距离估算力度
                    cue_pos = cue_ball.state.rvw[0]
                    target_pos = target_ball.state.rvw[0]
                    pocket_pos = pocket.center
                    
                    dist_cue_to_target = np.linalg.norm(target_pos - cue_pos)
                    dist_target_to_pocket = np.linalg.norm(pocket_pos - target_pos)
                    total_dist = dist_cue_to_target + dist_target_to_pocket
                    
                    # 力度估算公式：增加基础力度
                    # 台球桌长约2m，需要较大力度
                    base_v0 = 2.2 + total_dist * 1.8  # [优化] 略微降低力度系数，避免白球乱飞
                    cushions = sol.get('cushions', 0)
                    
                    # ==================== 4档力度选择 ====================
                    # 根据距离和库数确定合适的力度档位范围
                    # 距离分档: <0.8m 近距离, 0.8-1.5m 中距离, >1.5m 远距离
                    if total_dist < 0.8:
                        # 近距离：小力、中力
                        suitable_powers = ['soft', 'medium']
                    elif total_dist < 1.5:
                        # 中距离：中力、大力
                        suitable_powers = ['medium', 'hard']
                    else:
                        # 远距离：大力、极大力
                        suitable_powers = ['hard', 'very_hard']
                    
                    # 如果有库边，增加一档力度 (最大到 very_hard)
                    if cushions > 0:
                        power_upgrade = {
                            'soft': 'medium',
                            'medium': 'hard',
                            'hard': 'very_hard',
                            'very_hard': 'very_hard'
                        }
                        suitable_powers = list(dict.fromkeys([power_upgrade.get(p, p) for p in suitable_powers]))
                    
                    
                    for power_name in suitable_powers:
                        v0 = self.power_levels[power_name]
                        
                        action = {
                            'V0': v0,
                            'phi': sol['phi'],
                            'theta': 0,
                            'a': 0,
                            'b': 0
                        }
                        
                        all_candidates.append({
                            'action': action,
                            'target_id': target_id,
                            'pocket_id': pocket_id,
                            'shot_type': sol['type'],
                            'cut_angle': sol.get('cut_angle', 0),
                            'cushions': cushions,
                            'power_level': power_name,
                            'solution': sol
                        })
        
        if not all_candidates:
            return []
        
        print(f"  [阶段1] 共生成 {len(all_candidates)} 个候选方案")
        
        # ==================== 阶段1: 无噪声模拟验证 ====================
        print(f"  [阶段1] 无噪声模拟验证中...")
        
        verified_candidates = []
        
        for candidate in all_candidates:
            # 无噪声模拟一次 (使用统一的模拟函数)
            result = self.simulate_and_score(
                candidate['action'],
                balls,
                table,
                candidate['target_id'],
                legal_targets,
                num_simulations=1,
                add_noise=False
            )
            
            # 保存结果
            candidate['verify_success'] = result['is_success']
            candidate['verify_foul'] = result['is_foul']
            candidate['verify_position'] = result.get('position_score', 0)
            
            # 计算验证阶段评分
            if result['is_success'] and not result['is_foul']:
                # 成功进球且无犯规: 高分
                verify_score = 100 + result.get('position_score', 0)
            elif result['is_success'] and result['is_foul']:
                # 进球但犯规: 中分
                verify_score = 30
            elif not result['is_foul']:
                # 未进球但无犯规: 低分
                verify_score = 10
            else:
                # 犯规且未进球: 负分
                verify_score = -20
            
            # 切角惩罚
            verify_score -= candidate.get('cut_angle', 0) * 0.3
            # 库边惩罚 (Kick球比Bank球更不稳定)
            cushions = candidate.get('cushions', 0)
            if candidate.get('shot_type', '') == 'Kick':
                verify_score -= cushions * 15  # Kick球惩罚更重
            else:
                verify_score -= cushions * 10  # Bank球惩罚
            
            candidate['verify_score'] = verify_score
            
            # 只保留有效方案 (进球 或 无犯规)
            if result['is_success'] or not result['is_foul']:
                verified_candidates.append(candidate)
        
        print(f"  [阶段1] 有效方案: {len(verified_candidates)} 个")
        
        if not verified_candidates:
            # 无有效方案，选择评分最高的原始候选
            all_candidates.sort(key=lambda x: -x.get('verify_score', -100))
            verified_candidates = all_candidates[:8]
            print(f"  [阶段1] 无有效方案，使用评分 Top-10")
        
        # ==================== 阶段2: Top-15 精细模拟 ====================
        # 按验证评分排序
        verified_candidates.sort(key=lambda x: -x['verify_score'])
        top_candidates = verified_candidates[:15]
        
        # ==================== 致命失误过滤 ====================
        # 在精细模拟前检查是否存在致命风险（白球+黑8同落、非法黑8等）
        # 快速检查只需要 3 次模拟
        safe_candidates = []
        fatal_count = 0
        
        for candidate in top_candidates:
            fatal_result = self._check_fatal_failure(
                candidate['action'], balls, legal_targets, table, num_checks=3
            )
            
            if fatal_result['is_fatal'] and fatal_result['fatal_rate'] >= 0.3:
                # 致命风险超过 30%，跳过
                fatal_count += 1
                candidate['is_fatal'] = True
                candidate['fatal_type'] = fatal_result['fatal_type']
            else:
                candidate['is_fatal'] = False
                safe_candidates.append(candidate)
        
        if fatal_count > 0:
            print(f"  [致命检查] 过滤 {fatal_count} 个高风险方案 (黑8相关风险)")
        
        # 如果所有方案都有致命风险，保留风险最低的几个
        if not safe_candidates:
            print(f"  [致命检查] 警告：所有方案均有致命风险，保留风险最低的方案")
            # 按致命率排序，选择风险最低的
            for candidate in top_candidates:
                if 'fatal_rate' not in candidate:
                    candidate['fatal_rate'] = 1.0
            top_candidates.sort(key=lambda x: x.get('fatal_rate', 1.0))
            safe_candidates = top_candidates[:5]  # 保留风险最低的5个
        
        print(f"  [阶段2] 精细模拟 Top-{len(safe_candidates)} 候选 (15次):")
        
        scored_candidates = []
        for idx, candidate in enumerate(safe_candidates):
            # 15次精细模拟
            score_result = self.simulate_and_score(
                candidate['action'],
                balls,
                table,
                candidate['target_id'],
                legal_targets,
                num_simulations=15
            )
            
            candidate.update(score_result)
            scored_candidates.append(candidate)
            
            # 输出调试信息
            action = candidate['action']
            cushions = candidate.get('cushions', 0)
            cushion_str = f"{cushions}库" if cushions > 0 else "直球"
            
            print(f"    [{idx+1}] 目标={candidate['target_id']} -> {candidate['pocket_id']} "
                  f"| 类型={candidate['shot_type']}({cushion_str}) "
                  f"| 力度={candidate['power_level']}({action['V0']:.1f}) phi={action['phi']:.1f}° "
                  f"| 成功率={score_result['success_rate']:.0%} 犯规率={score_result['foul_rate']:.0%} "
                  f"| 走位={score_result['position_score']:.1f} "
                  f"| 总分={score_result['final_score']:.1f}")
        
        # 按 final_score 排序
        scored_candidates.sort(key=lambda x: -x['final_score'])
        return scored_candidates

    # ==================== 蒙特卡洛模拟 (统一版) ====================
    
    def simulate_and_score(self, action, balls, table, target_id, my_targets, 
                           num_simulations=None, add_noise=True):
        """
        统一的蒙特卡洛模拟函数
        
        参数:
            action: 击球动作
            balls: 球状态
            table: 球桌
            target_id: 目标球ID
            my_targets: 己方目标球列表
            num_simulations: 模拟次数，默认使用 self.num_simulation
            add_noise: 是否添加噪声，False 用于阶段1验证，True 用于阶段2精细模拟
        
        返回:
            dict: {
                'success_rate': 进球成功率,
                'foul_rate': 犯规率,
                'position_score': 走位平均分,
                'final_score': 综合评分,
                'is_success': 是否进球 (单次模拟时),
                'is_foul': 是否犯规 (单次模拟时)
            }
        """
        if num_simulations is None:
            num_simulations = self.num_simulation
            
        success_count = 0
        foul_count = 0
        position_scores = []
        penalty_total = 0
        last_result = None  # 保存最后一次结果（用于单次模拟返回详细信息）
        
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        
        for i in range(num_simulations):
            # 1. 是否添加噪声
            if add_noise:
                sim_action = self._add_noise(action)
            else:
                sim_action = action
            
            # 2. 物理模拟
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            balls_state_before = {bid: ball.state.s for bid, ball in sim_balls.items()}
            
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            shot.cue.set_state(**sim_action)
            
            # 带超时保护（无噪声模式用 simulate_with_timeout，有噪声模式直接模拟以加速）
            if not add_noise:
                if not simulate_with_timeout(shot, timeout=3):
                    last_result = {'is_success': False, 'is_foul': True, 'position_score': 0, 'penalty': 0}
                    foul_count += 1
                    continue
            else:
                try:
                    pt.simulate(shot, inplace=True)
                except Exception:
                    foul_count += 1
                    continue
            
            # 3. 分析结果
            result = self._analyze_shot_result(shot, balls_state_before, target_id, my_targets)
            last_result = result
            
            # 4. 走位评分 (已禁用)
            # 用户要求删除走位分
            result['position_score'] = 0
            position_scores.append(0)
            
            # 5. 统计
            if result['is_success']:
                success_count += 1
            if result['is_foul']:
                foul_count += 1
                penalty_total += result['penalty']
        
        # 汇总
        success_rate = success_count / num_simulations
        foul_rate = foul_count / num_simulations
        avg_position = np.mean(position_scores) if position_scores else 0
        avg_penalty = penalty_total / num_simulations
        
        # 综合评分公式
        if avg_penalty <= -50:  # 严重犯规
            if penalty_total <= -500:
                final_score = -1000  # 致命
            else:
                final_score = (
                    success_rate * 120 - foul_rate * 35 
                    + avg_penalty * 0.7
                )
        else:
            final_score = (
                success_rate * 120 - foul_rate * 35 
                + avg_penalty * 0.2
            )
        
        result_dict = {
            'success_rate': success_rate,
            'foul_rate': foul_rate,
            'position_score': avg_position,
            'final_score': final_score
        }
        
        # 单次模拟时附加详细信息
        if num_simulations == 1 and last_result:
            result_dict['is_success'] = last_result.get('is_success', False)
            result_dict['is_foul'] = last_result.get('is_foul', False)
        
        return result_dict

    def _add_noise(self, action):
        """为动作添加高斯噪声"""
        return {
            'V0': np.clip(action['V0'] + np.random.normal(0, self.noise_std['V0']), 0.5, 8.0),
            'phi': (action['phi'] + np.random.normal(0, self.noise_std['phi'])) % 360,
            'theta': np.clip(action.get('theta', 0) + np.random.normal(0, self.noise_std['theta']), 0, 90),
            'a': np.clip(action.get('a', 0) + np.random.normal(0, self.noise_std['a']), -0.5, 0.5),
            'b': np.clip(action.get('b', 0) + np.random.normal(0, self.noise_std['b']), -0.5, 0.5)
        }

    # ==================== CMA-ES 小范围微调 ====================
    
    def _cma_es_refine(self, geo_action, balls, my_targets, table, target_id):
        """
        使用 CMA-ES 在几何解附近进行小范围微调
        
        CMA-ES 优势：
        - 不需要梯度信息
        - 自适应搜索范围
        - 适合连续优化问题
        - 比贝叶斯更快收敛
        
        参数:
            geo_action: 几何求解得到的初始动作 {'V0', 'phi', 'theta', 'a', 'b'}
            balls: 球状态
            my_targets: 己方目标球
            table: 球桌
            target_id: 目标球ID
            
        返回:
            (refined_action, score): 微调后的动作和分数
        """
        if not self.CMA_ES_ENABLE:
            return geo_action, 0
        
        # 定义搜索边界
        bounds = np.array([
            [max(0.5, geo_action['V0'] - 1.5), min(8.0, geo_action['V0'] + 1.5)],  # V0
            [geo_action['phi'] - 15, geo_action['phi'] + 15],                       # phi
            [0, 12],                                                                 # theta
            [-0.25, 0.25],                                                          # a
            [-0.25, 0.25]                                                           # b
        ])
        
        # 归一化函数
        def normalize(x):
            return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        
        def denormalize(x):
            return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])
        
        def objective(x_norm):
            """目标函数：返回负分数（CMA-ES 最小化）"""
            try:
                x = denormalize(np.clip(x_norm, 0, 1))
                action = {
                    'V0': float(x[0]),
                    'phi': float(x[1]) % 360,
                    'theta': float(x[2]),
                    'a': float(x[3]),
                    'b': float(x[4])
                }
                
                # 使用统一的模拟函数（单次无噪声）
                result = self.simulate_and_score(
                    action, balls, table, target_id, my_targets,
                    num_simulations=1, add_noise=False
                )
                
                # 计算分数
                score = 0
                if result.get('is_success', result['success_rate'] > 0.5):
                    score += 80
                    # score += result['position_score'] * 0.5  # 禁用走位分
                
                if result.get('is_foul', result['foul_rate'] > 0.5):
                    # 使用 final_score 中的惩罚信息
                    score = min(score, result['final_score'])
                
                return -score  # CMA-ES 最小化
                
            except Exception:
                return 200  # 异常惩罚
        
        try:
            # 初始点（归一化）
            x0_norm = normalize(np.array([
                geo_action['V0'],
                geo_action['phi'],
                geo_action.get('theta', 0),
                geo_action.get('a', 0),
                geo_action.get('b', 0)
            ]))
            
            # CMA-ES 配置
            opts = {
                'bounds': [[0] * 5, [1] * 5],
                'maxiter': self.CMA_ES_MAXITER,
                'popsize': self.CMA_ES_POPSIZE,
                'verb_disp': 0,   # 静默模式
                'verb_log': 0,
                'verbose': -9,   # 完全静默
            }
            
            # 运行 CMA-ES
            es = cma.CMAEvolutionStrategy(x0_norm, 0.2, opts)
            es.optimize(objective)
            
            # 获取最优解
            best_x = denormalize(np.clip(es.result.xbest, 0, 1))
            best_score = -es.result.fbest  # 转回正分数
            
            refined_action = {
                'V0': float(best_x[0]),
                'phi': float(best_x[1]) % 360,
                'theta': float(best_x[2]),
                'a': float(best_x[3]),
                'b': float(best_x[4])
            }
            
            return refined_action, best_score
            
        except Exception as e:
            print(f"  [CMA-ES微调] 失败: {e}")
            return geo_action, 0

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
        
        # 7. 首球接触犯规 - 先碰到对方球或黑八（非打黑八时）
        elif first_contact:
            # 确定对方目标球
            if my_targets and my_targets[0] in [str(i) for i in range(1, 8)]:
                opponent_targets = [str(i) for i in range(9, 16)]
            else:
                opponent_targets = [str(i) for i in range(1, 8)]
            
            # 首球接触对方球
            if first_contact in opponent_targets:
                is_foul = True
                penalty = -80  # 首球犯规
            
            # 首球接触黑八（但目标不是黑八）
            elif first_contact == '8' and target_id != '8':
                is_foul = True
                penalty = -120  # 首球碰黑八更严重
        
        # 8. 未碰库
        if not is_foul and not new_pocketed and not (cue_hit_cushion or target_hit_cushion):
            is_foul = True
            penalty = -30
        
        # 白球最终位置
        cue_final_pos = shot.balls['cue'].state.rvw[0].copy()
        
        # === 关键修正：如果犯规，则通过视为失败 ===
        # 即使目标球进了，只要犯规（如先碰对手球）也算失败
        if is_foul:
            is_success = False
        
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
        统一的走位评分函数
        
        评估维度:
        1. 遍历所有剩余目标球和6个袋口，找最佳进攻机会
        2. 对每个(目标球, 袋口)组合评分：
           - 目标球到袋口距离 (越近越好)
           - 母球到目标球距离 (适中最好)
           - 击球角度 (越接近直球越好)
           - 路径障碍检测 (有遮挡扣分)
        3. 贴库惩罚
        4. 中心区域奖励
        
        返回:
            float: 走位分数 (0-100)
        """
        R = 0.028575  # 球半径
        
        # 确保 cue_pos 是 3D
        if len(cue_pos) == 2:
            cue_pos = np.array([cue_pos[0], cue_pos[1], 0])
        cue_xy = cue_pos[:2]
        
        # 1. 计算剩余目标球
        remaining = [t for t in my_targets if t in balls_after and balls_after[t].state.s != 4]
        if not remaining:
            remaining = ['8']  # 清台后打黑8
        
        # 2. 遍历所有(目标球, 袋口)组合，计算最高分
        best_score = 0.0
        
        for target_id in remaining:
            if target_id not in balls_after:
                continue
            target_ball = balls_after[target_id]
            if target_ball.state.s == 4:  # 已进袋
                continue
            
            target_pos = target_ball.state.rvw[0]
            target_xy = target_pos[:2]
            
            # 遍历所有袋口
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = get_pocket_position(pocket)
                pocket_xy = pocket_pos[:2]
                
                # === 评分维度 ===
                
                # 2.1 目标球到袋口距离 (满分15分)
                dist_target_to_pocket = np.linalg.norm(target_xy - pocket_xy)
                dist_pocket_score = 15 * (1 - min(dist_target_to_pocket / 2.5, 1.0))
                
                # 2.2 母球到目标球距离 (满分15分，0.3-0.8m最佳)
                dist_cue_to_target = np.linalg.norm(cue_xy - target_xy)
                if dist_cue_to_target < 0.1:
                    dist_cue_score = 2  # 贴球
                elif dist_cue_to_target < 0.3:
                    dist_cue_score = 7 + 5 * ((dist_cue_to_target - 0.1) / 0.2)
                elif dist_cue_to_target <= 0.8:
                    dist_cue_score = 15  # 最佳
                elif dist_cue_to_target < 1.5:
                    dist_cue_score = 15 - 8 * ((dist_cue_to_target - 0.8) / 0.7)
                else:
                    dist_cue_score = 5  # 太远
                
                # 2.3 击球角度 (满分20分，180度=直球最佳)
                vec_cue_to_target = target_xy - cue_xy
                vec_target_to_pocket = pocket_xy - target_xy
                angle = calculate_angle(vec_cue_to_target, vec_target_to_pocket)
                angle_score = 20 * (angle / 180.0)
                
                # 2.4 路径障碍检测
                obstacle_penalty = 0
                for bid, ball in balls_after.items():
                    if bid in ['cue', target_id] or ball.state.s == 4:
                        continue
                    ball_pos = ball.state.rvw[0][:2]
                    
                    # 母球到目标球被挡
                    if point_to_segment_distance(ball_pos, cue_xy, target_xy) < 2 * R + 0.01:
                        obstacle_penalty -= 30
                        break
                    # 目标球到袋口被挡
                    if point_to_segment_distance(ball_pos, target_xy, pocket_xy) < 2 * R + 0.01:
                        obstacle_penalty -= 15
                
                # 单组合得分
                combo_score = dist_pocket_score + dist_cue_score + angle_score + obstacle_penalty
                best_score = max(best_score, combo_score)
        
        # 3. 贴库惩罚
        dist_to_rail = min(
            cue_pos[0] - R, table.w - R - cue_pos[0],
            cue_pos[1] - R, table.l - R - cue_pos[1]
        )
        if dist_to_rail < 0.03:
            rail_penalty = -25
        elif dist_to_rail < 0.08:
            rail_penalty = -10
        else:
            rail_penalty = 0
        
        # 4. 中心区域奖励
        center = np.array([table.w / 2, table.l / 2, 0])
        dist_to_center = np.linalg.norm(cue_pos - center)
        max_dist = np.sqrt((table.w / 2) ** 2 + (table.l / 2) ** 2)
        if dist_to_center < max_dist * 0.3:
            center_bonus = 15
        elif dist_to_center < max_dist * 0.5:
            center_bonus = 5
        else:
            center_bonus = 0
        
        # 总分 = 基础分(0-50) + 贴库(-25~0) + 中心(0~15) + 偏移(25)
        total_score = best_score + rail_penalty + center_bonus + 25
        
        return max(0, total_score)


    # 库边反弹物理参数
    # 恢复系数 (垂直分量): 约 0.85-0.90
    # 平行摩擦系数: 约 0.86-0.90
    CUSHION_RESTITUTION = 0.87  # 垂直方向恢复系数
    CUSHION_FRICTION = 0.88     # 平行方向速度保持系数
    
    @staticmethod
    def get_mirror(point, rail):
        """理想镜像（入射角=反射角）"""
        p = point.copy()
        p[rail['axis']] = 2 * rail['val'] - p[rail['axis']]
        return p
    
    @staticmethod
    def get_adjusted_mirror(point, start_pos, rail, e=0.87, mu=0.88):
        """
        考虑弹性损耗的调整镜像
        
        物理原理：
        - 垂直分量: v_n_out = e * v_n_in (恢复系数)
        - 平行分量: v_t_out = mu * v_t_in (摩擦保持)
        
        由于平行分量损耗比垂直分量小，反射角会比入射角更"陡峭"
        
        调整策略：
        镜像点需要向库边方向拉近，以补偿平行速度损耗
        """
        p = point.copy()
        
        # 计算到库边的距离
        dist_to_rail = abs(point[rail['axis']] - rail['val'])
        
        # 计算平行方向的距离
        other_axis = 1 - rail['axis']
        parallel_dist = abs(point[other_axis] - start_pos[other_axis])
        
        # 调整系数：平行速度损耗导致镜像点需要调整
        # 实际反射角 tan(θ_out) = (mu * v_parallel) / (e * v_normal)
        # 相对于理想反射: adjustment = e / mu
        adjustment = e / mu  # ≈ 0.99，即镜像点需要稍微拉近
        
        # 调整镜像点
        p[rail['axis']] = 2 * rail['val'] - point[rail['axis']]
        
        # 微调：将镜像点向库边拉近一点
        # 这会使出射角更陡峭（更垂直于库边）
        mirror_dist = abs(p[rail['axis']] - rail['val'])
        new_dist = mirror_dist * adjustment
        
        if p[rail['axis']] > rail['val']:
            p[rail['axis']] = rail['val'] + new_dist
        else:
            p[rail['axis']] = rail['val'] - new_dist
        
        return p
        
    @staticmethod
    def get_cushion_path(start_pos, end_pos, rail_sequence, use_physics=True):
        """
        计算库边反弹路径
        
        参数:
            start_pos: 起始位置
            end_pos: 目标位置
            rail_sequence: 库边序列
            use_physics: 是否考虑弹性损耗 (默认True)
        """
        e = NewAgent.CUSHION_RESTITUTION
        mu = NewAgent.CUSHION_FRICTION
        
        # 1. 从后往前生成镜像目标点
        mirrored_targets = []
        current_target = end_pos
        prev_pos = end_pos  # 用于计算调整镜像
        
        for rail in reversed(rail_sequence):
            if use_physics:
                current_target = NewAgent.get_adjusted_mirror(current_target, prev_pos, rail, e, mu)
            else:
                current_target = NewAgent.get_mirror(current_target, rail)
            mirrored_targets.append(current_target)
            prev_pos = current_target
        
        # 2. 从前往后计算交点
        targets_to_aim = mirrored_targets[::-1]
        path_points = [start_pos]
        current_p = start_pos
        
        for i, rail in enumerate(rail_sequence):
            target_to_aim = targets_to_aim[i]
            
            vec = target_to_aim - current_p
            vec[2] = 0
            
            if abs(vec[rail['axis']]) < 1e-6: return None
            t = (rail['val'] - current_p[rail['axis']]) / vec[rail['axis']]
            
            if t <= 1e-4: return None
            
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
                
                # 增加 2mm 的安全余量，防止擦边导致的物理碰撞误差
                if dist < 2 * R + 0.002:
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

            # 2. 踢球 (Kick Hit to Ghost) - 限制最多1库
            max_kick_cushions = 1  # 只搜索1库，2库以上成功率太低
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
             
        # A2. Target Bank - 限制最多1库
        max_bank_cushions = 1  # 只搜索1库翻袋，2库以上成功率太低
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
                    
                    # ==================== 过滤条件：只保留最多1库的方案 ====================
                    # Kick-Bank (1库Kick + 1库Bank = 2库) 也不要
                    if total_cushions >= 2:
                        continue  # 跳过2库及以上的方案
                    
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
