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


# ============ 超时安全机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass


def _timeout_handler(signum, frame):
    raise SimulationTimeoutError("物理模拟超时")


def safe_simulate(shot, timeout=3):
    """
    带超时保护的物理模拟
    
    Returns:
        bool: 模拟是否成功完成
    """
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


# ============ 奖励计算 ============
def calculate_shot_reward(shot: pt.System, initial_state: dict, player_targets: list):
    """
    计算击球结果的奖励分数
    
    考虑因素:
    - 进球情况（己方/对方/白球/黑8）
    - 首球碰撞合法性
    - 碰库规则
    """
    # 识别新进袋的球
    newly_pocketed = [
        bid for bid, ball in shot.balls.items() 
        if ball.state.s == 4 and initial_state[bid].state.s != 4
    ]
    
    own_pocketed = [bid for bid in newly_pocketed if bid in player_targets]
    enemy_pocketed = [
        bid for bid in newly_pocketed 
        if bid not in player_targets and bid not in ["cue", "8"]
    ]
    
    cue_pocketed = "cue" in newly_pocketed
    eight_pocketed = "8" in newly_pocketed

    # 分析首球碰撞
    first_contact = _find_first_ball_contact(shot.events)
    foul_first_hit = _check_first_hit_foul(first_contact, player_targets, initial_state)
    
    # 分析碰库
    cue_hit_rail, target_hit_rail = _check_rail_contact(shot.events, first_contact)
    foul_no_rail = (
        len(newly_pocketed) == 0 and 
        first_contact is not None and 
        not cue_hit_rail and not target_hit_rail
    )
        
    # 计算总分
    score = 0
    
    # 致命犯规
    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_legal_eight = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 150 if is_legal_eight else -500
    
    # 规则犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
    
    # 进球得分
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 基础分（合法击球但未进球）
    if score == 0 and not any([cue_pocketed, eight_pocketed, foul_first_hit, foul_no_rail]):
        score = 10
        
    return score


def _find_first_ball_contact(events):
    """找到白球首次接触的目标球"""
    valid_ball_ids = {str(i) for i in range(1, 16)}
    
    for event in events:
        event_type = str(event.event_type).lower()
        ball_ids = list(event.ids) if hasattr(event, 'ids') else []
        
        # 跳过库边和袋口事件
        if 'cushion' in event_type or 'pocket' in event_type:
            continue
        
        if 'cue' in ball_ids:
            other_balls = [bid for bid in ball_ids if bid != 'cue' and bid in valid_ball_ids]
            if other_balls:
                return other_balls[0]
    
    return None


def _check_first_hit_foul(first_contact, player_targets, initial_state):
    """检查首球碰撞是否犯规"""
    # 未碰到任何球
    if first_contact is None:
        # 只剩黑8时允许不碰球（特殊情况）
        if len(initial_state) > 2 or player_targets != ['8']:
            return True
        return False
    
    # 碰到的不是目标球
    return first_contact not in player_targets


def _check_rail_contact(events, first_contact_ball):
    """检查白球和首次接触球是否碰库"""
    cue_hit_rail = False
    target_hit_rail = False
    
    for event in events:
        event_type = str(event.event_type).lower()
        if 'cushion' not in event_type:
            continue
        
        ball_ids = list(event.ids) if hasattr(event, 'ids') else []
        
        if 'cue' in ball_ids:
            cue_hit_rail = True
        if first_contact_ball and first_contact_ball in ball_ids:
            target_hit_rail = True
    
    return cue_hit_rail, target_hit_rail


class NewAgent(Agent):
    """
    台球AI Agent - 重构版
    
    核心功能:
    1. 精确几何路径计算（多库反弹）
    2. 分层决策流程
    3. CMA-ES参数优化
    4. 鲁棒性验证
    5. 防守策略评估
    """
    
    def __init__(self):
        super().__init__()
        
        # 物理常量
        self.BALL_RADIUS = 0.028575
        self.TABLE_WIDTH = 1.9812
        self.TABLE_HEIGHT = 0.9906
        
        # 库边坐标
        self.rails = {
            'x_pos': self.TABLE_WIDTH / 2,
            'x_neg': -self.TABLE_WIDTH / 2,
            'y_pos': self.TABLE_HEIGHT / 2,
            'y_neg': -self.TABLE_HEIGHT / 2
        }
        
        # 执行噪声参数
        self.execution_noise = {
            'V0': 0.1, 'phi': 0.1, 'theta': 0.1, 'a': 0.003, 'b': 0.003
        }
        
        # 力度档位（备选）
        self.power_presets = {
            'soft': 2.5,
            'medium': 4.0,
            'hard': 5.5,
            'very_hard': 7.0,
        }
        
        # CMA-ES优化配置
        self.cma_config = {
            'enabled': True,
            'max_iterations': 5,
            'population_size': 6,
        }
        
        # 鲁棒性验证配置
        self.robustness_config = {
            'enabled': True,
            'num_trials': 5,
        }
        
        print("[NewAgent] 重构版初始化完成")

    # ============================================================================
    # 基础工具
    # ============================================================================
    
    def get_safe_action(self):
        """返回安全的默认动作（原地不动）"""
        return {'V0': 0, 'phi': 0, 'theta': 0, 'a': 0, 'b': 0}
    
    def calculate_2d_distance(self, pos1, pos2):
        """计算两点间的2D距离"""
        return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
    
    def normalize_vector(self, vec):
        """归一化2D向量"""
        vec = np.array(vec[:2])
        norm = np.linalg.norm(vec)
        return np.array([1.0, 0.0]) if norm < 1e-6 else vec / norm
    
    def vector_to_angle(self, direction_vec):
        """将方向向量转换为角度 [0, 360)"""
        phi = np.arctan2(direction_vec[1], direction_vec[0]) * 180 / np.pi
        return phi % 360
    
    def get_nearest_pocket_distance(self, ball_pos, table):
        """获取球到最近袋口的距离"""
        return min(
            self.calculate_2d_distance(ball_pos, pocket.center) 
            for pocket in table.pockets.values()
        )
    
    def apply_execution_noise(self, action):
        """模拟执行误差"""
        noise = self.execution_noise
        return {
            'V0': np.clip(action['V0'] + np.random.normal(0, noise['V0']), 0.5, 8.0),
            'phi': (action['phi'] + np.random.normal(0, noise['phi'])) % 360,
            'theta': np.clip(action['theta'] + np.random.normal(0, noise['theta']), 0, 90),
            'a': np.clip(action['a'] + np.random.normal(0, noise['a']), -0.5, 0.5),
            'b': np.clip(action['b'] + np.random.normal(0, noise['b']), -0.5, 0.5)
        }

    # ============================================================================
    # 几何路径计算
    # ============================================================================
    
    @staticmethod
    def compute_multi_rail_path(start_pos, end_pos, rail_sequence):
        """
        计算多库反弹路径（镜像法）
        
        Args:
            start_pos: 起点 (x, y, z)
            end_pos: 终点 (x, y, z)
            rail_sequence: 库边序列 [{'name': 'x_pos', 'val': 0.99}, ...]
        
        Returns:
            list of np.ndarray: 路径点 [start, bounce1, ..., end]
            None: 无解
        """
        if not rail_sequence:
            return [np.array(start_pos), np.array(end_pos)]
        
        # 镜像终点
        mirrored_end = np.array(end_pos[:2])
        for rail in reversed(rail_sequence):
            axis = 0 if 'x' in rail['name'] else 1
            mirrored_end[axis] = 2 * rail['val'] - mirrored_end[axis]
        
        # 计算直线路径到镜像终点
        path = [np.array([start_pos[0], start_pos[1], 0])]
        current_pos = np.array(start_pos[:2])
        direction = mirrored_end - current_pos
        
        if np.linalg.norm(direction) < 1e-6:
            return None
        
        direction = direction / np.linalg.norm(direction)
        
        # 计算每个反弹点
        for rail in rail_sequence:
            axis = 0 if 'x' in rail['name'] else 1
            
            if abs(direction[axis]) < 1e-6:
                return None
            
            t = (rail['val'] - current_pos[axis]) / direction[axis]
            if t <= 0:
                return None
            
            bounce_point = current_pos + t * direction
            path.append(np.array([bounce_point[0], bounce_point[1], 0]))
            
            # 反射
            direction[axis] = -direction[axis]
            current_pos = bounce_point
        
        path.append(np.array([end_pos[0], end_pos[1], 0]))
        return path

    # ============================================================================
    # 走位评估
    # ============================================================================

    def extract_pocket_position(self, pocket):
        """提取袋口位置（含中袋修正）"""
        if hasattr(pocket, 'center'):
            pos = np.array([pocket.center[0], pocket.center[1], 0])
        elif hasattr(pocket, 'a'):
            pos = np.array([pocket.a, pocket.b, 0])
        else:
            pos = np.array([pocket.x, pocket.y, 0])
        
        # 中袋修正
        pocket_id = getattr(pocket, 'id', '')
        if 'c' in pocket_id.lower():
            side_rail_width = 0.0645
            offset = side_rail_width * 0.5
            pos[0] += offset if pos[0] < 0.5 else -offset
        
        return pos

    def compute_angle_between_vectors(self, v1, v2):
        """计算两向量夹角（度）"""
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-9 or v2_norm < 1e-9:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))

    def compute_point_to_segment_distance(self, point, seg_start, seg_end):
        """计算点到线段的最短距离"""
        point = np.array(point[:2])
        seg_start = np.array(seg_start[:2])
        seg_end = np.array(seg_end[:2])
        
        segment_vec = seg_end - seg_start
        point_vec = point - seg_start
        
        dot_product = np.dot(point_vec, segment_vec)
        
        if dot_product <= 0:
            return np.linalg.norm(point - seg_start)
        
        segment_length_sq = np.dot(segment_vec, segment_vec)
        
        if segment_length_sq <= dot_product:
            return np.linalg.norm(point - seg_end)
        
        projection = seg_start + (dot_product / segment_length_sq) * segment_vec
        return np.linalg.norm(point - projection)

    def evaluate_cue_ball_position(self, cue_pos, my_targets, balls_state, table):
        """
        评估白球位置的优劣（走位评分）
        
        考虑:
        - 到下一目标的距离和角度
        - 是否有障碍球
        - 是否贴库
        - 是否在台面中心
        """
        R = self.BALL_RADIUS
        cue_pos = np.array(cue_pos)
        cue_xy = cue_pos[:2]
        
        # 确定剩余目标
        remaining_targets = [
            tid for tid in my_targets 
            if tid in balls_state and balls_state[tid].state.s != 4
        ]
        if not remaining_targets:
            remaining_targets = ['8']
        
        best_score = 0.0
        
        # 评估每个可能的下一杆
        for target_id in remaining_targets:
            if target_id not in balls_state or balls_state[target_id].state.s == 4:
                continue
            
            target_pos = balls_state[target_id].state.rvw[0]
            target_xy = target_pos[:2]
            
            for pocket in table.pockets.values():
                pocket_pos = self.extract_pocket_position(pocket)
                pocket_xy = pocket_pos[:2]
                
                # 距离评分
                target_to_pocket_dist = np.linalg.norm(target_xy - pocket_xy)
                pocket_distance_score = 15 * (1 - min(target_to_pocket_dist / 2.5, 1.0))
                
                cue_to_target_dist = np.linalg.norm(cue_xy - target_xy)
                if cue_to_target_dist < 0.1:
                    cue_distance_score = 2
                elif cue_to_target_dist < 0.3:
                    cue_distance_score = 7 + 5 * ((cue_to_target_dist - 0.1) / 0.2)
                elif cue_to_target_dist <= 0.8:
                    cue_distance_score = 15
                elif cue_to_target_dist < 1.5:
                    cue_distance_score = 15 - 8 * ((cue_to_target_dist - 0.8) / 0.7)
                else:
                    cue_distance_score = 5
                
                # 角度评分
                vec_cue_to_target = target_xy - cue_xy
                vec_target_to_pocket = pocket_xy - target_xy
                angle = self.compute_angle_between_vectors(vec_cue_to_target, vec_target_to_pocket)
                angle_score = 20 * (angle / 180.0)
                
                # 障碍惩罚
                obstacle_penalty = 0
                for bid, ball in balls_state.items():
                    if bid in ['cue', target_id] or ball.state.s == 4:
                        continue
                    
                    ball_pos = ball.state.rvw[0]
                    
                    # 白球到目标球路径障碍
                    if self.compute_point_to_segment_distance(ball_pos, cue_xy, target_xy) < 2 * R + 0.01:
                        obstacle_penalty -= 30
                        break
                    
                    # 目标球到袋口路径障碍
                    if self.compute_point_to_segment_distance(ball_pos, target_xy, pocket_xy) < 2 * R + 0.01:
                        obstacle_penalty -= 15
                
                best_score = max(
                    best_score, 
                    pocket_distance_score + cue_distance_score + angle_score + obstacle_penalty
                )
        
        # 贴库惩罚
        distance_to_rail = min(
            cue_pos[0] - R, 
            self.TABLE_WIDTH - R - cue_pos[0],
            cue_pos[1] - R, 
            self.TABLE_HEIGHT - R - cue_pos[1]
        )
        rail_penalty = -25 if distance_to_rail < 0.03 else (-10 if distance_to_rail < 0.08 else 0)
        
        # 中心区域奖励
        table_center = np.array([self.TABLE_WIDTH/2, self.TABLE_HEIGHT/2, 0])
        distance_to_center = np.linalg.norm(cue_pos - table_center)
        max_distance = np.sqrt((self.TABLE_WIDTH/2)**2 + (self.TABLE_HEIGHT/2)**2)
        
        center_bonus = 15 if distance_to_center < max_distance * 0.3 else (
            5 if distance_to_center < max_distance * 0.5 else 0
        )
        
        return max(0, best_score + rail_penalty + center_bonus + 25)

    def compute_enhanced_reward(self, shot, initial_state, player_targets, table):
        """
        增强奖励函数
        
        = 基础奖励 + 黑8安全 + 走位评分
        """
        base_reward = calculate_shot_reward(shot, initial_state, player_targets)
        
        targeting_eight = (player_targets == ['8'])
        
        # 黑8保护（未清台时）
        if not targeting_eight and '8' in shot.balls and shot.balls['8'].state.s != 4:
            eight_dist_before = self.get_nearest_pocket_distance(
                initial_state['8'].state.rvw[0], table
            )
            eight_dist_after = self.get_nearest_pocket_distance(
                shot.balls['8'].state.rvw[0], table
            )
            
            if eight_dist_after < eight_dist_before:
                base_reward -= (eight_dist_before - eight_dist_after) * 150
        
        # 走位评分
        if 'cue' in shot.balls and shot.balls['cue'].state.s != 4:
            cue_pos = shot.balls['cue'].state.rvw[0]
            position_score = self.evaluate_cue_ball_position(
                cue_pos, player_targets, shot.balls, table
            )
            base_reward += position_score * 0.15
        
        return base_reward

    # ============================================================================
    # 模拟与评估
    # ============================================================================
    
    def simulate_and_evaluate(self, action, num_simulations, balls, my_targets, table, 
                             early_stop_threshold=20, apply_noise=False):
        """
        模拟动作并返回平均奖励
        
        Args:
            action: 击球参数
            num_simulations: 模拟次数
            early_stop_threshold: 早停阈值
            apply_noise: 是否添加执行噪声
        """
        rewards = []
        
        try:
            for trial_idx in range(num_simulations):
                # 深拷贝状态
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                shot = pt.System(
                    table=sim_table, 
                    balls=sim_balls, 
                    cue=pt.Cue(cue_ball_id="cue")
                )
                
                # 设置动作
                if apply_noise:
                    noisy_action = self.apply_execution_noise(action)
                    shot.cue.set_state(**noisy_action)
                else:
                    shot.cue.set_state(**action)
                
                # 模拟
                if not safe_simulate(shot, timeout=3):
                    rewards.append(-100)
                    continue
                
                # 评估
                trial_reward = self.compute_enhanced_reward(
                    shot, balls, my_targets, sim_table
                )
                rewards.append(trial_reward)
                
                # 早停
                if trial_reward < early_stop_threshold and len(rewards) > 1:
                    return float(np.mean(rewards))
            
            return float(np.mean(rewards))
            
        except Exception:
            return -999

    def check_fatal_foul_risk(self, action, balls, my_targets, table, 
                             num_tests=10, acceptable_risk=0.1):
        """
        检测致命犯规风险（白球落袋、黑8误进等）
        
        Returns:
            (fatal_rate, fatal_count, avg_reward)
        """
        targeting_eight = (my_targets == ['8'])
        fatal_count = 0
        error_count = 0
        rewards = []
        
        for trial_idx in range(num_tests):
            try:
                # 早停（如果风险已经明显过高）
                if fatal_count / (trial_idx + 1) > acceptable_risk + 0.1:
                    break
                
                # 模拟
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                shot = pt.System(
                    table=sim_table, 
                    balls=sim_balls,
                    cue=pt.Cue(cue_ball_id="cue")
                )
                
                noisy_action = self.apply_execution_noise(action)
                shot.cue.set_state(**noisy_action)
                
                if not safe_simulate(shot, timeout=3):
                    error_count += 1
                    continue
                
                rewards.append(
                    self.compute_enhanced_reward(shot, balls, my_targets, sim_table)
                )
                
                # 检测致命失误
                newly_pocketed = [
                    bid for bid, ball in shot.balls.items()
                    if ball.state.s == 4 and balls[bid].state.s != 4
                ]
                
                if "cue" in newly_pocketed and "8" in newly_pocketed:
                    fatal_count += 1
                elif "8" in newly_pocketed and not targeting_eight:
                    fatal_count += 1
                    
            except Exception:
                error_count += 1
        
        successful_tests = num_tests - error_count
        if successful_tests == 0:
            return 1.0, num_tests, -999
        
        fatal_rate = fatal_count / successful_tests
        avg_reward = float(np.mean(rewards)) if rewards else -500
        
        return fatal_rate, fatal_count, avg_reward

    # ============================================================================
    # 几何计算
    # ============================================================================
    
    def compute_ghost_ball_position(self, target_pos, pocket_pos):
        """计算瞄球点（Ghost Ball）"""
        direction = self.normalize_vector(
            np.array(pocket_pos[:2]) - np.array(target_pos[:2])
        )
        return np.array(target_pos[:2]) - direction * (2 * self.BALL_RADIUS)
    
    def compute_cut_angle(self, cue_pos, target_pos, pocket_pos):
        """计算切球角度"""
        ghost_pos = self.compute_ghost_ball_position(target_pos, pocket_pos)
        
        vec1 = self.normalize_vector(ghost_pos - np.array(cue_pos[:2]))
        vec2 = self.normalize_vector(np.array(pocket_pos[:2]) - np.array(target_pos[:2]))
        
        angle1 = np.arctan2(vec1[1], vec1[0])
        angle2 = np.arctan2(vec2[1], vec2[0])
        angle_diff = (angle2 - angle1) * 180 / np.pi
        
        # 归一化到 (-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff <= -180:
            angle_diff += 360
        
        return angle_diff
    
    def compute_english_for_cut(self, cut_angle):
        """根据切球角度计算塞球参数（侧旋）"""
        sign = 1 if cut_angle <= 0 else -1
        magnitude = min(0.5, abs(cut_angle) / 180 * 0.5)
        return sign * magnitude
    
    def create_basic_shot(self, cue_pos, target_pos, pocket_pos):
        """生成基础几何击球参数"""
        ghost_pos = self.compute_ghost_ball_position(target_pos, pocket_pos)
        direction = self.normalize_vector(ghost_pos - np.array(cue_pos[:2]))
        phi = self.vector_to_angle(direction)
        
        distance = self.calculate_2d_distance(cue_pos, ghost_pos)
        
        # 自适应力度
        if distance < 0.8:
            power = min(2.0 + distance * 1.5, 7.5)
        else:
            power = min(4.0 + distance * 0.8, 7.5)
        
        return {
            'V0': float(power),
            'phi': float(phi),
            'theta': 0.0,
            'a': 0.0,
            'b': 0.0
        }
    
    def create_bank_shot(self, cue_pos, target_pos, pocket_pos, rail_id):
        """生成翻袋击球参数（一库）"""
        mirrored_pocket = np.array(pocket_pos[:2])
        
        if 'x' in rail_id:
            mirrored_pocket[0] = 2 * self.rails[rail_id] - mirrored_pocket[0]
        else:
            mirrored_pocket[1] = 2 * self.rails[rail_id] - mirrored_pocket[1]
        
        return self.create_basic_shot(cue_pos, target_pos, mirrored_pocket)

    # ============================================================================
    # 路径分析
    # ============================================================================
    
    def count_path_obstacles(self, balls, start_pos, end_pos, exclude_ball_ids=['cue']):
        """计算路径上的障碍球数量"""
        obstacle_count = 0
        path_vector = np.array(end_pos[:2]) - np.array(start_pos[:2])
        path_length = np.linalg.norm(path_vector)
        
        if path_length < 1e-6:
            return 0
        
        path_direction = path_vector / path_length
        
        for ball_id, ball in balls.items():
            if ball_id in exclude_ball_ids or ball.state.s == 4:
                continue
            
            ball_to_start = ball.state.rvw[0][:2] - np.array(start_pos[:2])
            projection_length = np.dot(ball_to_start, path_direction)
            
            # 检查球是否在路径投影范围内
            if 0 < projection_length < path_length:
                projection_point = np.array(start_pos[:2]) + path_direction * projection_length
                perpendicular_distance = np.linalg.norm(
                    ball.state.rvw[0][:2] - projection_point
                )
                
                if perpendicular_distance < self.BALL_RADIUS * 2.2:
                    obstacle_count += 1
        
        return obstacle_count

    # ============================================================================
    # 目标选择
    # ============================================================================
    
    def rank_shot_options(self, balls, my_targets, table, 
                         max_options=3, defensive_mode=False):
        """
        评估并排序所有可能的击球方案
        
        Returns:
            list: 按得分排序的击球方案
        """
        all_options = []
        cue_pos = balls['cue'].state.rvw[0]
        
        for target_id in my_targets:
            if balls[target_id].state.s == 4:
                continue
            
            target_pos = balls[target_id].state.rvw[0]
            
            # 遍历每个袋口
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # 路径障碍检测
                cue_to_target_obstacles = self.count_path_obstacles(
                    balls, cue_pos, target_pos, ['cue', target_id]
                )
                target_to_pocket_obstacles = self.count_path_obstacles(
                    balls, target_pos, pocket_pos, ['cue', target_id]
                )
                
                # 几何参数
                total_distance = (
                    self.calculate_2d_distance(cue_pos, target_pos) + 
                    self.calculate_2d_distance(target_pos, pocket_pos)
                )
                cut_angle = self.compute_cut_angle(cue_pos, target_pos, pocket_pos)
                
                # 基础得分
                score = 100 - (total_distance * 20 + abs(cut_angle) * 0.5)
                
                # 黑8安全惩罚
                if target_id != '8' and '8' in balls and balls['8'].state.s != 4:
                    eight_distance = self.calculate_2d_distance(
                        target_pos, balls['8'].state.rvw[0]
                    )
                    if eight_distance < 0.3:
                        score -= (0.3 - eight_distance) * 150

                # 直接进球方案
                if cue_to_target_obstacles == 0 and target_to_pocket_obstacles == 0:
                    all_options.append({
                        'type': 'direct',
                        'target_id': target_id,
                        'pocket_id': pocket_id,
                        'score': score,
                        'cut_angle': cut_angle
                    })

                # 绕障方案（塞球）
                elif cue_to_target_obstacles <= 1 and target_to_pocket_obstacles <= 1:
                    curve_score = score - 20 - abs(cut_angle) * 0.1
                    all_options.append({
                        'type': 'curve',
                        'target_id': target_id,
                        'pocket_id': pocket_id,
                        'cut_angle': cut_angle,
                        'score': curve_score
                    })
                
                # 防守模式下跳过翻袋
                if defensive_mode:
                    continue
                
                # 翻袋方案
                for rail_id in self.rails.keys():
                    mirrored_pocket = np.array(pocket_pos[:2])
                    axis_idx = 0 if 'x' in rail_id else 1
                    mirrored_pocket[axis_idx] = 2 * self.rails[rail_id] - mirrored_pocket[axis_idx]
                    
                    cue_clear = self.count_path_obstacles(
                        balls, cue_pos, target_pos, ['cue', target_id]
                    ) == 0
                    bank_clear = self.count_path_obstacles(
                        balls, target_pos, mirrored_pocket, ['cue', target_id]
                    ) == 0
                    
                    if cue_clear and bank_clear:
                        bank_distance = (
                            self.calculate_2d_distance(cue_pos, target_pos) + 
                            self.calculate_2d_distance(target_pos, mirrored_pocket)
                        )
                        bank_cut_angle = self.compute_cut_angle(
                            cue_pos, target_pos, mirrored_pocket
                        )
                        
                        bank_score = 100 - (bank_distance * 25 + abs(bank_cut_angle) * 0.6 + 40)
                        
                        if bank_score > 0:
                            all_options.append({
                                'type': 'bank',
                                'target_id': target_id,
                                'pocket_id': pocket_id,
                                'rail_id': rail_id,
                                'score': bank_score,
                                'cut_angle': bank_cut_angle
                            })
        
        all_options.sort(key=lambda x: x['score'], reverse=True)
        return all_options[:max_options]

    # ============================================================================
    # CMA-ES优化
    # ============================================================================
    
    def optimize_with_cma_es(self, initial_action, balls, my_targets, table, 
                            targeting_black_eight=False):
        """
        使用CMA-ES优化击球参数，并进行鲁棒性验证
        
        Returns:
            (optimized_action, verified_reward)
        """
        
        # 参数搜索范围
        param_bounds = np.array([
            [max(0.5, initial_action['V0'] - 1.5), min(8.0, initial_action['V0'] + 1.5)],
            [initial_action['phi'] - 15, initial_action['phi'] + 15],
            [0, 10],
            [max(initial_action['a'] - 0.2, -0.5), min(0.5, initial_action['a'] + 0.2)],
            [max(initial_action['b'] - 0.2, -0.5), min(0.5, initial_action['b'] + 0.2)]
        ])
        
        def normalize_params(x):
            return (x - param_bounds[:, 0]) / (param_bounds[:, 1] - param_bounds[:, 0])
        
        def denormalize_params(x):
            return param_bounds[:, 0] + x * (param_bounds[:, 1] - param_bounds[:, 0])
        
        # 初始点
        initial_params = np.array([
            initial_action['V0'], initial_action['phi'], 0, 
            initial_action['a'], initial_action['b']
        ])
        x0_normalized = normalize_params(initial_params)
        
        # CMA-ES配置
        cma_options = {
            'bounds': [[0]*5, [1]*5],
            'maxiter': self.cma_config['max_iterations'],
            'popsize': self.cma_config['population_size'] if targeting_black_eight else 4,
            'verb_disp': 0,
            'verb_log': 0
        }
        
        def objective_function(x_normalized):
            x = denormalize_params(np.clip(x_normalized, 0, 1))
            action = {
                'V0': float(x[0]),
                'phi': float(x[1]),
                'theta': float(x[2]),
                'a': float(x[3]),
                'b': float(x[4])
            }
            return -self.simulate_and_evaluate(
                action, 2, balls, my_targets, table, 20, True
            )
        
        try:
            # 运行CMA-ES
            es = cma.CMAEvolutionStrategy(x0_normalized, 0.2, cma_options)
            es.optimize(objective_function)
            
            # 提取最优解
            best_params = denormalize_params(np.clip(es.result.xbest, 0, 1))
            optimized_action = {
                'V0': float(best_params[0]),
                'phi': float(best_params[1]),
                'theta': float(best_params[2]),
                'a': float(best_params[3]),
                'b': float(best_params[4])
            }
            theoretical_reward = -es.result.fbest
            
            # 鲁棒性验证
            if self.robustness_config['enabled']:
                verified_reward = self.simulate_and_evaluate(
                    optimized_action, 
                    self.robustness_config['num_trials'], 
                    balls, my_targets, table, 20, True
                )
                
                # 对比原始方案
                original_reward = self.simulate_and_evaluate(
                    initial_action, 2, balls, my_targets, table, 20, True
                )
                
                if verified_reward > original_reward:
                    return optimized_action, verified_reward
                else:
                    return initial_action, original_reward
            
            return optimized_action, theoretical_reward
            
        except Exception:
            return None, -999

    # ============================================================================
    # 防守策略
    # ============================================================================
    
    def get_opponent_targets(self, my_targets):
        """获取对手的目标球"""
        all_numbered_balls = set(str(i) for i in range(1, 16))
        my_ball_set = set(my_targets)
        opponent_balls = all_numbered_balls - my_ball_set - {'8'}
        return list(opponent_balls)
    
    def evaluate_defensive_quality(self, action, balls, my_targets, table):
        """
        评估防守效果（对手回球难度）
        
        Returns:
            float: 对手最佳机会得分（越低表示防守越好）
        """
        # 模拟击球
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        shot = pt.System(
            table=sim_table, 
            balls=sim_balls,
            cue=pt.Cue(cue_ball_id="cue")
        )
        
        shot.cue.set_state(**action)
        
        if not safe_simulate(shot, timeout=3):
            return 999
        
        # 检查是否犯规
        safety_reward = calculate_shot_reward(shot, balls, my_targets)
        if safety_reward < 0:
            return 999 - safety_reward
        
        # 评估对手最佳回球机会
        opponent_targets = self.get_opponent_targets(my_targets)
        opponent_best_options = self.rank_shot_options(
            shot.balls, opponent_targets, shot.table, 
            max_options=1, defensive_mode=True
        )
        
        if not opponent_best_options:
            return -100  # 完美防守
        
        return opponent_best_options[0]['score']
    
    def find_best_defensive_shot(self, balls, my_targets, table):
        """
        寻找最佳防守方案
        
        策略优先级:
        1. 大力出奇迹（Force Attack）
        2. 斯诺克/贴库防守
        """
        
        # ========================================================================
        # 策略1: 大力出奇迹
        # ========================================================================
        print("[Agent] 尝试大力出奇迹...")
        miracle_shots = []
        cue_pos = balls['cue'].state.rvw[0]
        
        for target_id in my_targets:
            if balls[target_id].state.s == 4:
                continue
            
            target_pos = balls[target_id].state.rvw[0]
            
            # 必须能直接打到球
            if self.count_path_obstacles(balls, cue_pos, target_pos, ['cue', target_id]) > 0:
                continue
            
            for pocket in table.pockets.values():
                pocket_pos = pocket.center
                
                # 生成大力击球
                action = self.create_basic_shot(cue_pos, target_pos, pocket_pos)
                action['V0'] = 7.5  # 最大力度
                
                # 过滤极端切角
                cut_angle = self.compute_cut_angle(cue_pos, target_pos, pocket_pos)
                if abs(cut_angle) > 80:
                    continue
                
                # 几何评分
                total_distance = (
                    self.calculate_2d_distance(cue_pos, target_pos) + 
                    self.calculate_2d_distance(target_pos, pocket_pos)
                )
                geo_score = 100 - (total_distance * 10 + abs(cut_angle) * 0.8)
                
                miracle_shots.append({
                    'action': action,
                    'geo_score': geo_score,
                    'description': f"miracle {target_id}->pocket"
                })
        
        # 尝试Top-5奇迹球
        miracle_shots.sort(key=lambda x: x['geo_score'], reverse=True)
        
        for candidate in miracle_shots[:5]:
            # 初步模拟
            sim_reward = self.simulate_and_evaluate(
                candidate['action'], 
                trials=2, 
                balls=balls, 
                my_targets=my_targets, 
                table=table,
                early_stop_threshold=15, 
                apply_noise=True
            )
            
            if sim_reward > 40:
                # 安全性检查
                fatal_rate, _, verified_reward = self.check_fatal_foul_risk(
                    candidate['action'], balls, my_targets, table, num_tests=5
                )
                
                if verified_reward > 40 and fatal_rate <= 0.1:
                    print(f"[Agent] 奇迹生效！{candidate['description']} "
                          f"分数:{verified_reward:.1f} 风险:{fatal_rate:.1%}")
                    return candidate['action']

        print("[Agent] 奇迹未发生，转入常规防守...")

        # ========================================================================
        # 策略2: 常规防守
        # ========================================================================
        print("[Agent] 计算防守方案...")
        
        defensive_options = []
        
        # 子策略A: 藏在己方球后
        for my_ball_id in my_targets:
            if balls[my_ball_id].state.s == 4:
                continue
            
            my_ball_pos = balls[my_ball_id].state.rvw[0]
            
            for angle_deg in [0, 90, 180, 270]:
                angle_rad = angle_deg * np.pi / 180
                hiding_position = my_ball_pos[:2] + np.array([
                    np.cos(angle_rad), np.sin(angle_rad)
                ]) * (self.BALL_RADIUS * 3)
                
                direction = self.normalize_vector(hiding_position - cue_pos[:2])
                phi = self.vector_to_angle(direction)
                
                defensive_options.append({
                    'V0': 0.8,
                    'phi': phi,
                    'theta': 0,
                    'a': 0,
                    'b': 0
                })
        
        # 子策略B: 推己方球贴库
        for my_ball_id in my_targets:
            if balls[my_ball_id].state.s == 4:
                continue
            
            my_ball_pos = balls[my_ball_id].state.rvw[0]
            
            # 找最近的库边
            nearest_distance = float('inf')
            target_rail_position = None
            
            for rail_name, rail_value in self.rails.items():
                if 'x' in rail_name:
                    distance = abs(my_ball_pos[0] - rail_value)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        target_rail_position = [rail_value, my_ball_pos[1]]
                else:
                    distance = abs(my_ball_pos[1] - rail_value)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        target_rail_position = [my_ball_pos[0], rail_value]
            
            if target_rail_position:
                action = self.create_basic_shot(cue_pos, my_ball_pos, target_rail_position)
                action['V0'] = 1.0
                defensive_options.append(action)
        
        if not defensive_options:
            return self.get_safe_action()
        
        # 评估所有防守方案
        best_defensive_action = None
        lowest_opponent_score = float('inf')
        
        for action in defensive_options:
            opponent_best_score = self.evaluate_defensive_quality(
                action, balls, my_targets, table
            )
            
            if opponent_best_score < lowest_opponent_score:
                lowest_opponent_score = opponent_best_score
                best_defensive_action = action
        
        print(f"[Agent] 防守方案确定 - 对手最佳机会: {lowest_opponent_score:.1f}")
        
        return best_defensive_action

    # ============================================================================
    # 开球检测
    # ============================================================================
    
    def is_break_situation(self, balls):
        """判断是否为开球局面（三角架未散）"""
        my_ball_count = sum(
            1 for bid in balls 
            if bid not in ('cue', '8') and balls[bid].state.s != 4 and int(bid) >= 9
        )
        opponent_ball_count = sum(
            1 for bid in balls 
            if bid not in ('cue', '8') and balls[bid].state.s != 4 and int(bid) < 8
        )
        eight_ball_on_table = ('8' in balls and balls['8'].state.s != 4)
        
        return my_ball_count == 7 and opponent_ball_count == 7 and eight_ball_on_table

    def create_suicide_break_shot(self, balls, table):
        """
        生成自杀开球
        
        策略: 故意把白球打进袋，让对手获得球权来开球
        """
        cue_pos = balls['cue'].state.rvw[0][:2]
        
        # 找最近的袋口
        nearest_pocket = None
        min_distance = float('inf')
        
        for pocket in table.pockets.values():
            pocket_pos = pocket.center[:2]
            distance = np.linalg.norm(np.array(pocket_pos) - np.array(cue_pos))
            if distance < min_distance:
                min_distance = distance
                nearest_pocket = pocket_pos
        
        if nearest_pocket is not None:
            direction_vec = np.array(nearest_pocket) - np.array(cue_pos)
            phi = np.degrees(np.arctan2(direction_vec[1], direction_vec[0])) % 360
            
            print(f"[NewAgent] 自杀开球: 瞄准最近袋口 (phi={phi:.1f}°)")
            
            return {
                'V0': 3.0,
                'phi': phi,
                'theta': 0.0,
                'a': 0.0,
                'b': 0.0
            }
        
        return self.get_safe_action()

    # ============================================================================
    # 主决策逻辑
    # ============================================================================
    
    def decision(self, balls=None, my_targets=None, table=None):
        """
        主决策函数
        
        决策流程:
        1. 检查开球局面 -> 自杀开球
        2. 分层筛选目标
        3. 几何评估 + CMA-ES优化
        4. 鲁棒性验证
        5. 备选方案
        6. 防守兜底
        """
        if not all([balls, my_targets, table]):
            return self.get_safe_action()
        
        try:
            print("\n" + "="*60)
            print("[NewAgent] 开始决策...")
            
            # 检查开球
            if self.is_break_situation(balls):
                print("[NewAgent] 检测到开球局面，执行自杀策略")
                return self.create_suicide_break_shot(balls, table)
            
            # 确定目标球
            remaining_targets = [
                bid for bid in my_targets if balls[bid].state.s != 4
            ]
            if not remaining_targets and "8" in balls and balls["8"].state.s != 4:
                my_targets = ['8']
                print("[NewAgent] 切换到黑8")
            
            targeting_black_eight = (my_targets == ['8'])

            # 设置阈值
            high_quality_threshold = 120 if targeting_black_eight else 55
            acceptable_threshold = 120 if targeting_black_eight else 50
            safe_fatal_risk = 0.05

            # 第一层: 选择候选目标
            candidate_options = self.rank_shot_options(
                balls, my_targets, table, 
                max_options=4 if self.is_break_situation(balls) else 20
            )
            
            if not candidate_options:
                print("[NewAgent] 无可行目标，切换防守")
                return self.find_best_defensive_shot(balls, my_targets, table)

            # 第二层: 几何评估
            geometrically_evaluated = []
            cue_pos = balls['cue'].state.rvw[0]
            
            print(f"[NewAgent] 评估 {len(candidate_options)} 个候选方案")
            
            for option in candidate_options:
                target_pos = balls[option['target_id']].state.rvw[0]
                pocket_pos = table.pockets[option['pocket_id']].center
                
                # 生成基础动作
                if option['type'] == 'bank':
                    base_action = self.create_bank_shot(
                        cue_pos, target_pos, pocket_pos, option['rail_id']
                    )
                elif option['type'] == 'curve':
                    base_action = self.create_basic_shot(cue_pos, target_pos, pocket_pos)
                    base_action['a'] = self.compute_english_for_cut(option['cut_angle'])
                else:
                    base_action = self.create_basic_shot(cue_pos, target_pos, pocket_pos)
                
                # 快速评估
                geo_reward = self.simulate_and_evaluate(
                    base_action, 1, balls, my_targets, table, 30, False
                )
                
                if geo_reward > -100:
                    geometrically_evaluated.append((base_action, geo_reward, option))

            geometrically_evaluated.sort(key=lambda x: x[1], reverse=True)
            
            if geometrically_evaluated:
                print(f"[NewAgent] 最佳几何分数: {geometrically_evaluated[0][1]:.2f}")
            
            # 第三层: 精细优化Top-8
            backup_options = []
            
            for i, (base_action, geo_reward, option) in enumerate(geometrically_evaluated[:8]):
                action_candidate = base_action
                reward_candidate = geo_reward
                shot_description = f"{option['type']} {option['target_id']}->{option['pocket_id']}"
                
                if geo_reward > high_quality_threshold:
                    # 高分方案 -> 直接验证
                    fatal_rate, _, verified_reward = self.check_fatal_foul_risk(
                        action_candidate, balls, my_targets, table, num_tests=3
                    )
                    
                    if verified_reward >= high_quality_threshold and fatal_rate <= safe_fatal_risk:
                        # 最终确认
                        fatal_rate, _, verified_reward = self.check_fatal_foul_risk(
                            action_candidate, balls, my_targets, table, num_tests=12
                        )
                        if verified_reward >= high_quality_threshold and fatal_rate <= safe_fatal_risk:
                            print(f"[NewAgent] ✓ 采用高分方案: {shot_description}")
                            print(f"           分数={verified_reward:.2f}, 风险={fatal_rate:.1%}")
                            print("="*60 + "\n")
                            return action_candidate
                else:
                    # 低分方案 -> CMA-ES优化
                    optimized_action, optimized_reward = self.optimize_with_cma_es(
                        base_action, balls, my_targets, table, targeting_black_eight
                    )
                    
                    if optimized_action and optimized_reward > acceptable_threshold - 20:
                        action_candidate = optimized_action
                        reward_candidate = optimized_reward
                        
                        fatal_rate, _, verified_reward = self.check_fatal_foul_risk(
                            action_candidate, balls, my_targets, table, num_tests=15
                        )
                        
                        if verified_reward >= acceptable_threshold and fatal_rate <= safe_fatal_risk:
                            print(f"[NewAgent] ✓ 采用优化方案: {shot_description}")
                            print(f"           分数={verified_reward:.2f}, 风险={fatal_rate:.1%}")
                            print("="*60 + "\n")
                            return action_candidate
                        elif verified_reward > 0 and fatal_rate <= safe_fatal_risk:
                            backup_options.append(
                                (action_candidate, verified_reward, shot_description, fatal_rate)
                            )
            
            # 第四层: 备选方案
            print(f"[NewAgent] 从 {len(backup_options)} 个备选中选择")
            safe_backups = [opt for opt in backup_options if opt[3] <= safe_fatal_risk]
            
            if safe_backups:
                best_backup = max(safe_backups, key=lambda x: x[1])
                best_action, best_reward, best_description, _ = best_backup
                
                if best_reward > 10:
                    print(f"[NewAgent] 采用备选方案: {best_description} (分数={best_reward:.2f})")
                    print("="*60 + "\n")
                    return best_action

            # 最终: 防守
            print("[NewAgent] 无可行进攻，切换防守")
            print("="*60 + "\n")
            return self.find_best_defensive_shot(balls, my_targets, table)
            
        except Exception as e:
            print(f"[NewAgent] 决策异常: {e}")
            import traceback
            traceback.print_exc()
            return self.get_safe_action()