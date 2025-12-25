"""
融合版 NewAgent - 结合两个方案的优点

架构来自方案二（简洁高效）:
- 分层决策流程
- 完善的防守评估
- 黑8风险和白球安全评分

增强来自方案一:
- 精确的多库反弹路径求解 (get_cushion_path)
- CMA-ES 优化后的鲁棒性验证
- 4档力度系统作为备选
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime
import itertools

from .agent import Agent
import copy
import signal
import random
import cma


# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass


def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")


def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟"""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过")
        return False
    except Exception as e:
        signal.alarm(0)
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ============ 奖励函数 ============
def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数
    """
    new_pocketed = [bid for bid, b in shot.balls.items() 
                   if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed 
                     if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 首球碰撞分析
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', 
                     '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 碰库分析
    cue_hit_cushion = False
    target_hit_cushion = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id and first_contact_ball_id in ids:
                target_hit_cushion = True

    foul_no_rail = (len(new_pocketed) == 0 and 
                   first_contact_ball_id is not None and 
                   not cue_hit_cushion and not target_hit_cushion)
        
    # 计算分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_legal_eight = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 150 if is_legal_eight else -500
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if (score == 0 and not cue_pocketed and not eight_pocketed and 
        not foul_first_hit and not foul_no_rail):
        score = 10
        
    return score


class NewAgent(Agent):
    """
    融合版台球AI Agent
    
    核心特性:
    1. 精确的几何路径求解（支持Kick/Bank多库）
    2. 分层决策流程（高效筛选）
    3. CMA-ES优化 + 鲁棒性验证
    4. 完善的防守策略（评估对手回球）
    5. 黑8风险和白球安全评估
    """
    
    def __init__(self):
        super().__init__()
        
        # 球桌常量
        self.BALL_RADIUS = 0.028575
        self.TABLE_WIDTH = 1.9812
        self.TABLE_HEIGHT = 0.9906
        
        # 库边位置
        self.cushions = {
            'x_pos': self.TABLE_WIDTH / 2,
            'x_neg': -self.TABLE_WIDTH / 2,
            'y_pos': self.TABLE_HEIGHT / 2,
            'y_neg': -self.TABLE_HEIGHT / 2
        }
        
        # 噪声参数
        self.noise_std = {
            'V0': 0.1, 'phi': 0.1, 'theta': 0.1, 'a': 0.003, 'b': 0.003
        }
        
        # 力度系统（自适应 + 档位备选）
        self.power_levels = {
            'soft': 2.5,
            'medium': 4.0,
            'hard': 5.5,
            'very_hard': 7.0,
        }
        
        # CMA-ES 配置
        self.CMA_ES_ENABLE = True
        self.CMA_ES_MAXITER = 5
        self.CMA_ES_POPSIZE = 6
        
        # 鲁棒性验证配置（来自方案一）
        self.ROBUSTNESS_CHECK_ENABLED = True
        self.ROBUSTNESS_TRIALS = 5
        
        print("[NewAgent] 融合版初始化完成")

    # ============================================================================
    # 基础工具函数
    # ============================================================================
    
    def _safe_action(self):
        """安全的默认动作"""
        return {'V0': 0, 'phi': 0, 'theta': 0, 'a': 0, 'b': 0}
    
    def _calc_dist(self, pos1, pos2):
        """计算2D距离"""
        return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
    
    def _unit_vector(self, vec):
        """归一化向量"""
        vec = np.array(vec[:2])
        norm = np.linalg.norm(vec)
        return np.array([1.0, 0.0]) if norm < 1e-6 else vec / norm
    
    def _direction_to_degrees(self, direction_vec):
        """向量转角度 [0, 360)"""
        phi = np.arctan2(direction_vec[1], direction_vec[0]) * 180 / np.pi
        return phi % 360
    
    def _distance_to_nearest_pocket(self, ball_pos, table):
        """到最近袋口的距离"""
        return min(self._calc_dist(ball_pos, pocket.center) 
                   for pocket in table.pockets.values())
    
    def _add_noise(self, action):
        """给动作添加噪声"""
        noise = self.noise_std
        return {
            'V0': np.clip(action['V0'] + np.random.normal(0, noise['V0']), 0.5, 8.0),
            'phi': (action['phi'] + np.random.normal(0, noise['phi'])) % 360,
            'theta': np.clip(action['theta'] + np.random.normal(0, noise['theta']), 0, 90),
            'a': np.clip(action['a'] + np.random.normal(0, noise['a']), -0.5, 0.5),
            'b': np.clip(action['b'] + np.random.normal(0, noise['b']), -0.5, 0.5)
        }

    # ============================================================================
    # 精确几何路径求解（来自方案一）
    # ============================================================================
    
    @staticmethod
    def get_cushion_path(start_pos, end_pos, cushion_sequence):
        """
        计算经过多个库边反弹的路径
        
        参数:
            start_pos: 起始位置 (x, y, z)
            end_pos: 目标位置 (x, y, z)
            cushion_sequence: 库边序列，每个元素是 {'name': 'x_pos/x_neg/y_pos/y_neg', 'val': float}
        
        返回:
            list of np.ndarray: 路径点列表 [start, bounce1, bounce2, ..., end]
            None: 无解
        """
        if not cushion_sequence:
            return [np.array(start_pos), np.array(end_pos)]
        
        # 镜像法求解多库反弹
        mirrored_end = np.array(end_pos[:2])
        
        for cushion in reversed(cushion_sequence):
            if 'x' in cushion['name']:
                mirrored_end[0] = 2 * cushion['val'] - mirrored_end[0]
            else:
                mirrored_end[1] = 2 * cushion['val'] - mirrored_end[1]
        
        # 从起点到镜像终点的直线
        path_points = [np.array([start_pos[0], start_pos[1], 0])]
        current_pos = np.array(start_pos[:2])
        direction = mirrored_end - current_pos
        
        if np.linalg.norm(direction) < 1e-6:
            return None
        
        direction = direction / np.linalg.norm(direction)
        
        # 计算每个反弹点
        for cushion in cushion_sequence:
            if 'x' in cushion['name']:
                # 垂直库边
                if abs(direction[0]) < 1e-6:
                    return None
                t = (cushion['val'] - current_pos[0]) / direction[0]
                if t <= 0:
                    return None
                bounce_point = current_pos + t * direction
                # 反射方向
                direction[0] = -direction[0]
            else:
                # 水平库边
                if abs(direction[1]) < 1e-6:
                    return None
                t = (cushion['val'] - current_pos[1]) / direction[1]
                if t <= 0:
                    return None
                bounce_point = current_pos + t * direction
                direction[1] = -direction[1]
            
            path_points.append(np.array([bounce_point[0], bounce_point[1], 0]))
            current_pos = bounce_point
        
        path_points.append(np.array([end_pos[0], end_pos[1], 0]))
        return path_points

    # ============================================================================
    # 改进的评分函数（来自方案二，带黑8风险和白球安全）
    # ============================================================================
    
    # ============================================================================
    # 走位评分工具函数
    # ============================================================================

    def _get_pocket_position(self, pocket):
        """提取袋口位置（带偏移修正）"""
        if hasattr(pocket, 'center'):
            pos = np.array([pocket.center[0], pocket.center[1], 0])
        elif hasattr(pocket, 'a'):
            pos = np.array([pocket.a, pocket.b, 0])
        else:
            pos = np.array([pocket.x, pocket.y, 0])
        
        pocket_id = getattr(pocket, 'id', '')
        if 'c' in pocket_id.lower():
            r_side = 0.0645
            offset = r_side * 0.5
            if pos[0] < 0.5: pos[0] += offset
            else: pos[0] -= offset
        return pos

    def _calculate_angle(self, v1, v2):
        """计算向量夹角"""
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm < 1e-9 or v2_norm < 1e-9: return 0.0
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))

    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """点到线段距离"""
        point = np.array(point[:2])
        seg_start = np.array(seg_start[:2])
        seg_end = np.array(seg_end[:2])
        v = seg_end - seg_start
        w = point - seg_start
        c1 = np.dot(w, v)
        if c1 <= 0: return np.linalg.norm(point - seg_start)
        c2 = np.dot(v, v)
        if c2 <= c1: return np.linalg.norm(point - seg_end)
        return np.linalg.norm(point - (seg_start + c1 / c2 * v))

    def _evaluate_position(self, cue_pos, my_targets, balls_after, table):
        """走位评分：评估白球停留位置对下一杆的利弊"""
        R = self.BALL_RADIUS
        cue_pos = np.array(cue_pos)
        cue_xy = cue_pos[:2]
        
        # 1. 确定剩余目标
        remaining = [t for t in my_targets if t in balls_after and balls_after[t].state.s != 4]
        if not remaining: remaining = ['8']
        
        best_score = 0.0
        
        # 2. 遍历下一杆的最佳机会
        for target_id in remaining:
            if target_id not in balls_after: continue
            target_ball = balls_after[target_id]
            if target_ball.state.s == 4: continue
            
            target_pos = target_ball.state.rvw[0]
            target_xy = target_pos[:2]
            
            for pocket in table.pockets.values():
                pocket_pos = self._get_pocket_position(pocket)
                pocket_xy = pocket_pos[:2]
                
                # 距离评分
                dist_t2p = np.linalg.norm(target_xy - pocket_xy)
                score_dist_p = 15 * (1 - min(dist_t2p / 2.5, 1.0))
                
                dist_c2t = np.linalg.norm(cue_xy - target_xy)
                if dist_c2t < 0.1: score_dist_c = 2
                elif dist_c2t < 0.3: score_dist_c = 7 + 5 * ((dist_c2t - 0.1) / 0.2)
                elif dist_c2t <= 0.8: score_dist_c = 15
                elif dist_c2t < 1.5: score_dist_c = 15 - 8 * ((dist_c2t - 0.8) / 0.7)
                else: score_dist_c = 5
                
                # 角度评分
                vec_c2t = target_xy - cue_xy
                vec_t2p = pocket_xy - target_xy
                angle = self._calculate_angle(vec_c2t, vec_t2p)
                score_angle = 20 * (angle / 180.0)
                
                # 障碍扣分
                penalty_obs = 0
                for bid, ball in balls_after.items():
                    if bid in ['cue', target_id] or ball.state.s == 4: continue
                    b_pos = ball.state.rvw[0]
                    if self._point_to_segment_distance(b_pos, cue_xy, target_xy) < 2 * R + 0.01:
                        penalty_obs -= 30
                        break
                    if self._point_to_segment_distance(b_pos, target_xy, pocket_xy) < 2 * R + 0.01:
                        penalty_obs -= 15
                
                best_score = max(best_score, score_dist_p + score_dist_c + score_angle + penalty_obs)
        
        # 3. 贴库惩罚
        dist_to_rail = min(cue_pos[0] - R, self.TABLE_WIDTH - R - cue_pos[0],
                          cue_pos[1] - R, self.TABLE_HEIGHT - R - cue_pos[1])
        penalty_rail = -25 if dist_to_rail < 0.03 else (-10 if dist_to_rail < 0.08 else 0)
        
        # 4. 中心奖励
        center = np.array([self.TABLE_WIDTH/2, self.TABLE_HEIGHT/2, 0])
        dist_center = np.linalg.norm(cue_pos - center)
        max_dist = np.sqrt((self.TABLE_WIDTH/2)**2 + (self.TABLE_HEIGHT/2)**2)
        bonus_center = 15 if dist_center < max_dist * 0.3 else (5 if dist_center < max_dist * 0.5 else 0)
        
        return max(0, best_score + penalty_rail + bonus_center + 25)

    def _improved_reward_function(self, shot, last_state, player_targets, table):
        """增强的奖励函数：基础分 + 黑8保护 + 走位评分"""
        base_score = analyze_shot_for_reward(shot, last_state, player_targets)
        
        targeting_eight = (player_targets == ['8'])
        
        # 1. 黑8保护（未清台时惩罚黑8靠近袋口）
        if not targeting_eight and '8' in shot.balls and shot.balls['8'].state.s != 4:
            eight_before = self._distance_to_nearest_pocket(last_state['8'].state.rvw[0], table)
            eight_after = self._distance_to_nearest_pocket(shot.balls['8'].state.rvw[0], table)
            if eight_after < eight_before:
                base_score -= (eight_before - eight_after) * 150
        
        # 2. 走位评分（替代了原有的简单白球安全分）
        # 只有当白球在桌面上时才计算
        if 'cue' in shot.balls and shot.balls['cue'].state.s != 4:
            cue_pos = shot.balls['cue'].state.rvw[0]
            # 计算走位分 (0-100) 并缩放权重加入总分
            # 权重设为 0.5，避免喧宾夺主（进球本身是 50 分）
            position_score = self._evaluate_position(cue_pos, player_targets, shot.balls, table)
            base_score += position_score * 0.15
        
        return base_score

    # ============================================================================
    # 模拟评估函数
    # ============================================================================
    
    def _evaluate_action(self, action, trials, balls, my_targets, table, 
                        threshold=20, enable_noise=False):
        """模拟动作并返回平均分数"""
        scores = []
        
        try:
            for trial_num in range(trials):
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                shot = pt.System(table=sim_table, balls=sim_balls, 
                               cue=pt.Cue(cue_ball_id="cue"))
                
                if enable_noise:
                    noisy_action = self._add_noise(action)
                    shot.cue.set_state(**noisy_action)
                else:
                    shot.cue.set_state(**action)
                
                if not simulate_with_timeout(shot, timeout=3):
                    scores.append(-100)
                    continue
                
                trial_score = self._improved_reward_function(
                    shot, balls, my_targets, sim_table)
                scores.append(trial_score)
                
                # 早停
                if trial_score < threshold and len(scores) > 1:
                    return float(np.mean(scores))
            
            return float(np.mean(scores))
            
        except Exception:
            return -999

    def _check_fatal_failure(self, action, balls, my_targets, table, 
                            num_trials=10, fatal_threshold=0.1):
        """检测致命失误（白球+黑8落袋等）"""
        targeting_eight = (my_targets == ['8'])
        fatal_count = 0
        error_count = 0
        scores = []
        
        for trial in range(num_trials):
            try:
                if fatal_count / (trial + 1) > fatal_threshold + 0.1:
                    break
                
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                shot = pt.System(table=sim_table, balls=sim_balls,
                               cue=pt.Cue(cue_ball_id="cue"))
                
                noisy_action = self._add_noise(action)
                shot.cue.set_state(**noisy_action)
                
                if not simulate_with_timeout(shot, timeout=3):
                    error_count += 1
                    continue
                
                scores.append(self._improved_reward_function(
                    shot, balls, my_targets, sim_table))
                
                new_pocketed = [bid for bid, ball in shot.balls.items()
                               if ball.state.s == 4 and balls[bid].state.s != 4]
                
                if "cue" in new_pocketed and "8" in new_pocketed:
                    fatal_count += 1
                elif "8" in new_pocketed and not targeting_eight:
                    fatal_count += 1
                    
            except Exception:
                error_count += 1
        
        success_count = num_trials - error_count
        if success_count == 0:
            return 1.0, num_trials, -999
        
        fatal_rate = fatal_count / success_count
        avg_score = float(np.mean(scores)) if scores else -500
        
        return fatal_rate, fatal_count, avg_score

    # ============================================================================
    # 几何计算
    # ============================================================================
    
    def _calc_ghost_ball(self, target_pos, pocket_pos):
        """计算瞄球点（Ghost Ball）"""
        direction = self._unit_vector(
            np.array(pocket_pos[:2]) - np.array(target_pos[:2]))
        return np.array(target_pos[:2]) - direction * (2 * self.BALL_RADIUS)
    
    def _calculate_cut_angle(self, cue_pos, target_pos, pocket_pos):
        """计算切球角度"""
        ghost_pos = self._calc_ghost_ball(target_pos, pocket_pos)
        vec1 = self._unit_vector(ghost_pos - np.array(cue_pos[:2]))
        vec2 = self._unit_vector(np.array(pocket_pos[:2]) - np.array(target_pos[:2]))
        
        angle1 = np.arctan2(vec1[1], vec1[0])
        angle2 = np.arctan2(vec2[1], vec2[0])
        angle_diff = (angle2 - angle1) * 180 / np.pi
        
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff <= -180:
            angle_diff += 360
        
        return angle_diff
    
    def _calc_curve_a(self, cut_angle):
        """根据切球角度计算塞球参数"""
        sign = 1 if cut_angle <= 0 else -1
        magnitude = min(0.5, abs(cut_angle) / 180 * 0.5)
        return sign * magnitude
    
    def _geo_shot(self, cue_pos, target_pos, pocket_pos):
        """生成基础几何击球参数"""
        ghost_pos = self._calc_ghost_ball(target_pos, pocket_pos)
        direction = self._unit_vector(ghost_pos - np.array(cue_pos[:2]))
        phi = self._direction_to_degrees(direction)
        
        dist = self._calc_dist(cue_pos, ghost_pos)
        
        # 自适应力度
        if dist < 0.8:
            V0 = min(2.0 + dist * 1.5, 7.5)
        else:
            V0 = min(4.0 + dist * 0.8, 7.5)
        
        return {
            'V0': float(V0),
            'phi': float(phi),
            'theta': 0.0,
            'a': 0.0,
            'b': 0.0
        }
    
    def _geo_bank_shot(self, cue_pos, target_pos, pocket_pos, cushion_id):
        """生成翻袋击球参数"""
        mirrored_pocket = np.array(pocket_pos[:2])
        
        if 'x' in cushion_id:
            mirrored_pocket[0] = 2 * self.cushions[cushion_id] - mirrored_pocket[0]
        else:
            mirrored_pocket[1] = 2 * self.cushions[cushion_id] - mirrored_pocket[1]
        
        return self._geo_shot(cue_pos, target_pos, mirrored_pocket)

    # ============================================================================
    # 路径检测
    # ============================================================================
    
    def _count_obstructions(self, balls, from_pos, to_pos, exclude_ids=['cue']):
        """计算路径上的障碍球数量"""
        count = 0
        line_vec = np.array(to_pos[:2]) - np.array(from_pos[:2])
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 1e-6:
            return 0
        
        line_dir = line_vec / line_length
        
        for bid, ball in balls.items():
            if bid in exclude_ids or ball.state.s == 4:
                continue
            
            vec_to_ball = ball.state.rvw[0][:2] - np.array(from_pos[:2])
            proj_length = np.dot(vec_to_ball, line_dir)
            
            if 0 < proj_length < line_length:
                perp_dist = np.linalg.norm(
                    ball.state.rvw[0][:2] - 
                    (np.array(from_pos[:2]) + line_dir * proj_length)
                )
                
                if perp_dist < self.BALL_RADIUS * 2.2:
                    count += 1
        
        return count

    # ============================================================================
    # 目标选择
    # ============================================================================
    
    def _choose_top_targets(self, balls, my_targets, table, 
                           num_choices=3, is_defense=False):
        """评估所有可能的击球选项，返回最佳的N个"""
        all_choices = []
        cue_pos = balls['cue'].state.rvw[0]
        
        for target_id in my_targets:
            if balls[target_id].state.s == 4:
                continue
            
            target_pos = balls[target_id].state.rvw[0]
            
            # 直接进球
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                cue_to_target_obs = self._count_obstructions(
                    balls, cue_pos, target_pos, ['cue', target_id])
                target_to_pocket_obs = self._count_obstructions(
                    balls, target_pos, pocket_pos, ['cue', target_id])
                
                dist = (self._calc_dist(cue_pos, target_pos) + 
                        self._calc_dist(target_pos, pocket_pos))
                cut_angle = self._calculate_cut_angle(cue_pos, target_pos, pocket_pos)
                
                score = 100 - (dist * 20 + abs(cut_angle) * 0.5)
                
                # 惩罚靠近黑8的击球
                if target_id != '8' and '8' in balls and balls['8'].state.s != 4:
                    eight_dist = self._calc_dist(target_pos, balls['8'].state.rvw[0])
                    if eight_dist < 0.3:
                        score -= (0.3 - eight_dist) * 150

                if cue_to_target_obs == 0 and target_to_pocket_obs == 0:
                    all_choices.append({
                        'type': 'direct',
                        'target_id': target_id,
                        'pocket_id': pocket_id,
                        'score': score,
                        'cut_angle': cut_angle
                    })

                elif cue_to_target_obs <= 1 and target_to_pocket_obs <= 1:
                    curve_score = score - 20 - abs(cut_angle) * 0.1
                    all_choices.append({
                        'type': 'curve',
                        'target_id': target_id,
                        'pocket_id': pocket_id,
                        'cut_angle': cut_angle,
                        'score': curve_score
                    })
                
                if is_defense:
                    continue
                
                # 翻袋（Bank）
                for cushion_id in self.cushions.keys():
                    mirrored_pocket = np.array(pocket_pos[:2])
                    idx = 0 if 'x' in cushion_id else 1
                    mirrored_pocket[idx] = 2 * self.cushions[cushion_id] - mirrored_pocket[idx]
                    
                    cue_clear = self._count_obstructions(
                        balls, cue_pos, target_pos, ['cue', target_id]) == 0
                    bank_clear = self._count_obstructions(
                        balls, target_pos, mirrored_pocket, ['cue', target_id]) == 0
                    
                    if cue_clear and bank_clear:
                        bank_dist = (self._calc_dist(cue_pos, target_pos) + 
                                    self._calc_dist(target_pos, mirrored_pocket))
                        bank_cut_angle = self._calculate_cut_angle(
                            cue_pos, target_pos, mirrored_pocket)
                        
                        bank_score = 100 - (bank_dist * 25 + abs(bank_cut_angle) * 0.6 + 40)
                        
                        if bank_score > 0:
                            all_choices.append({
                                'type': 'bank',
                                'target_id': target_id,
                                'pocket_id': pocket_id,
                                'cushion_id': cushion_id,
                                'score': bank_score,
                                'cut_angle': bank_cut_angle
                            })
        
        all_choices.sort(key=lambda x: x['score'], reverse=True)
        return all_choices[:num_choices]

    # ============================================================================
    # CMA-ES 优化 + 鲁棒性验证（融合方案一的验证机制）
    # ============================================================================
    
    def _cma_es_optimized(self, geo_action, balls, my_targets, table, is_black_eight=False):
        """CMA-ES 优化 + 鲁棒性验证"""
        
        bounds = np.array([
            [max(0.5, geo_action['V0'] - 1.5), min(8.0, geo_action['V0'] + 1.5)],
            [geo_action['phi'] - 15, geo_action['phi'] + 15],
            [0, 10],
            [max(geo_action['a'] - 0.2, -0.5), min(0.5, geo_action['a'] + 0.2)],
            [max(geo_action['b'] - 0.2, -0.5), min(0.5, geo_action['b'] + 0.2)]
        ])
        
        def normalize(x):
            return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        
        def denormalize(x):
            return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])
        
        x0_norm = normalize(np.array([
            geo_action['V0'], geo_action['phi'], 0, geo_action['a'], geo_action['b']
        ]))
        
        opts = {
            'bounds': [[0]*5, [1]*5],
            'maxiter': self.CMA_ES_MAXITER,
            'popsize': self.CMA_ES_POPSIZE if is_black_eight else 4,
            'verb_disp': 0,
            'verb_log': 0
        }
        
        def objective(x_norm):
            x = denormalize(np.clip(x_norm, 0, 1))
            action = {
                'V0': float(x[0]),
                'phi': float(x[1]),
                'theta': float(x[2]),
                'a': float(x[3]),
                'b': float(x[4])
            }
            return -self._evaluate_action(action, 2, balls, my_targets, table, 20, True)
        
        try:
            es = cma.CMAEvolutionStrategy(x0_norm, 0.2, opts)
            es.optimize(objective)
            
            best_x = denormalize(np.clip(es.result.xbest, 0, 1))
            optimized_action = {
                'V0': float(best_x[0]),
                'phi': float(best_x[1]),
                'theta': float(best_x[2]),
                'a': float(best_x[3]),
                'b': float(best_x[4])
            }
            theoretical_score = -es.result.fbest
            
            # ========== 鲁棒性验证（来自方案一）==========
            if self.ROBUSTNESS_CHECK_ENABLED:
                verified_score = self._evaluate_action(
                    optimized_action, self.ROBUSTNESS_TRIALS, 
                    balls, my_targets, table, 20, True
                )
                
                # 只有验证分数也足够好才返回优化结果
                original_score = self._evaluate_action(
                    geo_action, 2, balls, my_targets, table, 20, True
                )
                
                if verified_score > original_score:
                    return optimized_action, verified_score
                else:
                    # 优化后反而变差，返回原始方案
                    return geo_action, original_score
            
            return optimized_action, theoretical_score
            
        except Exception:
            return None, -999

    # ============================================================================
    # 防守策略（来自方案二）
    # ============================================================================
    
    def _get_opponent_targets(self, my_targets):
        """获取对手的目标球"""
        all_balls = set(str(i) for i in range(1, 16))
        my_set = set(my_targets)
        opponent_set = all_balls - my_set - {'8'}
        return list(opponent_set)
    
    def _evaluate_safety_shot(self, action, balls, my_targets, table):
        """评估防守效果：让对手最难打"""
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        shot = pt.System(table=sim_table, balls=sim_balls,
                        cue=pt.Cue(cue_ball_id="cue"))
        
        shot.cue.set_state(**action)
        
        if not simulate_with_timeout(shot, timeout=3):
            return 999
        
        safety_reward = analyze_shot_for_reward(shot, balls, my_targets)
        if safety_reward < 0:
            return 999 - safety_reward
        
        opponent_targets = self._get_opponent_targets(my_targets)
        opponent_choices = self._choose_top_targets(
            shot.balls, opponent_targets, shot.table, num_choices=1, is_defense=True)
        
        if not opponent_choices:
            return -100  # 完美防守
        
        return opponent_choices[0]['score']
    
    def _find_best_safety_shot(self, balls, my_targets, table):
        """寻找最佳防守方案"""
        print("[Agent] 切换防守模式")
        
        candidate_safeties = []
        cue_pos = balls['cue'].state.rvw[0]
        
        # 策略1：把白球藏到己方球后面
        for my_ball_id in my_targets:
            if balls[my_ball_id].state.s == 4:
                continue
            
            my_ball_pos = balls[my_ball_id].state.rvw[0]
            
            for angle_deg in [0, 90, 180, 270]:
                rad = angle_deg * np.pi / 180
                hide_target = my_ball_pos[:2] + np.array([
                    np.cos(rad), np.sin(rad)
                ]) * (self.BALL_RADIUS * 3)
                
                direction = self._unit_vector(hide_target - cue_pos[:2])
                phi = self._direction_to_degrees(direction)
                
                candidate_safeties.append({
                    'V0': 0.8,
                    'phi': phi,
                    'theta': 0,
                    'a': 0,
                    'b': 0
                })
        
        # 策略2：轻推己方球到库边
        for my_ball_id in my_targets:
            if balls[my_ball_id].state.s == 4:
                continue
            
            my_ball_pos = balls[my_ball_id].state.rvw[0]
            
            nearest_dist = float('inf')
            target_rail = None
            
            for cushion_name, cushion_val in self.cushions.items():
                if 'x' in cushion_name:
                    dist = abs(my_ball_pos[0] - cushion_val)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        target_rail = [cushion_val, my_ball_pos[1]]
                else:
                    dist = abs(my_ball_pos[1] - cushion_val)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        target_rail = [my_ball_pos[0], cushion_val]
            
            if target_rail:
                action = self._geo_shot(cue_pos, my_ball_pos, target_rail)
                action['V0'] = 1.0
                candidate_safeties.append(action)
        
        if not candidate_safeties:
            return self._safe_action()
        
        # 评估所有防守方案
        best_safety = None
        lowest_opp_score = float('inf')
        
        for action in candidate_safeties:
            opp_score = self._evaluate_safety_shot(action, balls, my_targets, table)
            
            if opp_score < lowest_opp_score:
                lowest_opp_score = opp_score
                best_safety = action
        
        print(f"[Agent] 防守方案选定 - 对手最佳回球估计: {lowest_opp_score:.1f}")
        
        return best_safety

    # ============================================================================
    # 开球检测
    # ============================================================================
    
    def _is_opening_state(self, balls):
        """判断是否为开球局面"""
        my_count = sum(1 for bid in balls 
                      if bid not in ('cue', '8') and balls[bid].state.s != 4 and int(bid) >= 9)
        op_count = sum(1 for bid in balls 
                      if bid not in ('cue', '8') and balls[bid].state.s != 4 and int(bid) < 8)
        eight_on = ('8' in balls and balls['8'].state.s != 4)
        return my_count == 7 and op_count == 7 and eight_on

    def _generate_suicide_break(self, balls, table):
        """
        生成自杀开球动作：直接把白球打进最近的袋口
        策略：故意犯规（白球落袋），让对手获得球权（线后自由球或其他规则），由对手来开球
        """
        cue_pos = balls['cue'].state.rvw[0][:2]
        
        # 找最近的袋口
        best_pocket = None
        min_dist = float('inf')
        
        for pocket in table.pockets.values():
            pocket_pos = pocket.center[:2]
            dist = np.linalg.norm(np.array(pocket_pos) - np.array(cue_pos))
            if dist < min_dist:
                min_dist = dist
                best_pocket = pocket_pos
        
        if best_pocket is not None:
             vec = np.array(best_pocket) - np.array(cue_pos)
             phi = np.degrees(np.arctan2(vec[1], vec[0])) % 360
             
             print(f"[NewAgent] 自杀开球目标: 最近袋口 (phi={phi:.1f}°)")
             
             return {
                 'V0': 3.0,   # 足够进球即可
                 'phi': phi,
                 'theta': 0.0,
                 'a': 0.0,
                 'b': 0.0
             }
        return self._safe_action()

    # ============================================================================
    # 主决策入口
    # ============================================================================
    
    def decision(self, balls=None, my_targets=None, table=None):
        """主决策函数"""
        if not all([balls, my_targets, table]):
            return self._safe_action()
        
        try:
            print("\n" + "="*60)
            print("[NewAgent] 开始决策...")
            
            # 检查开球局面
            if self._is_opening_state(balls):
                print("[NewAgent] 检测到开球局面，执行‘自杀’策略：把白球打进袋，让对手开球")
                return self._generate_suicide_break(balls, table)
            
            # 确定目标球
            remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
            if not remaining and "8" in balls and balls["8"].state.s != 4:
                my_targets = ['8']
                print("[NewAgent] 切换到黑8")
            is_black_eight = (my_targets == ['8'])

            # 阈值设定
            GEO_THRESHOLD = 120 if is_black_eight else 55
            SCORE_THRESHOLD = 120 if is_black_eight else 50
            SAFE_FATAL_THRESHOLD = 0.05

            # 第一层：选择最佳目标
            top_choices = self._choose_top_targets(
                balls, my_targets, table, 
                num_choices=4 if self._is_opening_state(balls) else 20
            )
            
            if not top_choices:
                print("[NewAgent] 未找到可行目标，切换防守...")
                return self._find_best_safety_shot(balls, my_targets, table)

            # 第二层：几何评估 + CMA-ES优化
            all_evaluated = []
            cue_pos = balls['cue'].state.rvw[0]
            geo_evaluated = []
            
            print(f"[NewAgent] 评估 {len(top_choices)} 个候选方案")
            
            for choice in top_choices:
                target_pos = balls[choice['target_id']].state.rvw[0]
                pocket_pos = table.pockets[choice['pocket_id']].center
                
                if choice['type'] == 'bank':
                    base_action = self._geo_bank_shot(
                        cue_pos, target_pos, pocket_pos, choice['cushion_id'])
                elif choice['type'] == 'curve':
                    base_action = self._geo_shot(cue_pos, target_pos, pocket_pos)
                    base_action['a'] = self._calc_curve_a(choice['cut_angle'])
                else:
                    base_action = self._geo_shot(cue_pos, target_pos, pocket_pos)
                
                geo_score = self._evaluate_action(
                    base_action, 1, balls, my_targets, table, 30, False)
                
                if geo_score > -100:
                    geo_evaluated.append((base_action, geo_score, choice))

            geo_evaluated.sort(key=lambda x: x[1], reverse=True)
            
            if geo_evaluated:
                print(f"[NewAgent] 最佳几何分数: {geo_evaluated[0][1]:.2f}")
            
            # 第三层：精细评估 Top-8
            for i, geo_choice in enumerate(geo_evaluated[:8]):
                base_action, geo_score, choice = geo_choice
                action_to_check, score_to_check = base_action, geo_score
                shot_type_str = f"{choice['type']} {choice['target_id']}->{choice['pocket_id']}"
                
                if geo_score > GEO_THRESHOLD:
                    # 高分方案直接验证
                    fatal_rate, _, verified_score = self._check_fatal_failure(
                        action_to_check, balls, my_targets, table, num_trials=3)
                    
                    if verified_score >= GEO_THRESHOLD and fatal_rate <= SAFE_FATAL_THRESHOLD:
                        fatal_rate, _, verified_score = self._check_fatal_failure(
                            action_to_check, balls, my_targets, table, num_trials=12)
                        if verified_score >= GEO_THRESHOLD and fatal_rate <= SAFE_FATAL_THRESHOLD:
                            print(f"[NewAgent] ✓ 采用高分方案: {shot_type_str}")
                            print(f"           分数={verified_score:.2f}, 致命率={fatal_rate:.1%}")
                            print("="*60 + "\n")
                            return action_to_check
                else:
                    # 低分方案尝试CMA-ES优化
                    opt_action, opt_score = self._cma_es_optimized(
                        base_action, balls, my_targets, table, is_black_eight)
                    
                    if opt_action and opt_score > SCORE_THRESHOLD - 20:
                        action_to_check, score_to_check = opt_action, opt_score
                        fatal_rate, _, verified_score = self._check_fatal_failure(
                            action_to_check, balls, my_targets, table, num_trials=15)
                        
                        if verified_score >= SCORE_THRESHOLD and fatal_rate <= SAFE_FATAL_THRESHOLD:
                            print(f"[NewAgent] ✓ 采用优化方案: {shot_type_str}")
                            print(f"           分数={verified_score:.2f}, 致命率={fatal_rate:.1%}")
                            print("="*60 + "\n")
                            return action_to_check
                        elif verified_score > 0 and fatal_rate <= SAFE_FATAL_THRESHOLD:
                            all_evaluated.append(
                                (action_to_check, verified_score, shot_type_str, fatal_rate))
            
            # 第四层：备选方案
            print(f"[NewAgent] 从 {len(all_evaluated)} 个备选中选择")
            safe_candidates = [c for c in all_evaluated if c[3] <= SAFE_FATAL_THRESHOLD]
            
            if safe_candidates:
                best_action, best_score, best_type, _ = max(
                    safe_candidates, key=lambda x: x[1])
                
                if best_score > 10:
                    print(f"[NewAgent] 采用备选方案: {best_type} (分数={best_score:.2f})")
                    print("="*60 + "\n")
                    return best_action

            # 最终：防守
            print("[NewAgent] 无可行进攻方案，切换防守策略...")
            print("="*60 + "\n")
            return self._find_best_safety_shot(balls, my_targets, table)
            
        except Exception as e:
            print(f"[NewAgent] 决策异常: {e}")
            import traceback
            traceback.print_exc()
            return self._safe_action()