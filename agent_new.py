"""
agent_new.py - 重构后的 Agent 决策模块

架构设计:
├── 基础设施层 (Infrastructure)
│   ├── SimulationTimeoutError - 超时异常
│   ├── VirtualState / VirtualBall - 辅助类
│   └── simulate_with_timeout - 带超时保护的模拟
│
├── 模拟层 (Simulation)
│   ├── simulate_shot - 单次模拟
│   └── simulate_shot_batch - 批量模拟（带噪声）
│
├── 评分层 (Scoring)
│   ├── score_position - 白球停点评分
│   ├── score_shot_result - 单次击球结果评分
│   └── score_plan - 方案综合评分
│
├── 求解器层 (Solver)
│   ├── solve_direct_shot - Direct 方案求解
│   └── (其他求解器预留)
│
└── Agent 决策层 (Decision)
    ├── attack_strategy - 进攻策略
    ├── defense_strategy - 防守策略
    └── decision - 主决策入口
"""

import math
import signal
import copy
import numpy as np
import pooltool as pt
from typing import Dict, List, Tuple, Optional, Any


# ============================================================================
#                           基础设施层 (Infrastructure)
# ============================================================================

class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass


class VirtualState:
    """虚拟球状态，用于构建模拟场景"""
    
    def __init__(self, pos: Tuple[float, float, float]):
        """
        参数:
            pos: 球的位置 (x, y, z)
        """
        self.rvw = np.array([np.array(pos), np.zeros(3), np.zeros(3)])
        self.s = 1  # 静止状态


class VirtualBall:
    """虚拟球对象，用于构建模拟场景"""
    
    def __init__(self, ball_id: str, pos: Tuple[float, float, float], R: float = 0.028575):
        """
        参数:
            ball_id: 球的ID
            pos: 球的位置 (x, y, z)
            R: 球的半径，默认标准台球半径
        """
        self.id = ball_id
        self.state = VirtualState(pos)
        self.params = type('Params', (), {'R': R})()


class MockPocket:
    """模拟袋口对象，用于存储优化后的袋口坐标"""
    def __init__(self, x, y, pid, radius):
        self.center = np.array([x, y, 0.0])
        self.id = pid
        self.radius = radius


def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")


def simulate_with_timeout(shot: pt.System, timeout: int = 3) -> bool:
    """
    带超时保护的物理模拟
    
    参数:
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回:
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明:
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # TODO: 实现超时保护的模拟逻辑
    raise NotImplementedError


# ============================================================================
#                              模拟层 (Simulation)
# ============================================================================

def simulate_shot(
    action: Dict[str, float],
    balls: Dict[str, Any],
    table: Any
) -> Tuple[bool, Optional[pt.System]]:
    """
    执行单次击球模拟
    
    参数:
        action: 击球参数字典 {'V0': float, 'phi': float, 'theta': float, 'a': float, 'b': float}
        balls: 当前球状态字典 {ball_id: Ball}
        table: 球桌对象
    
    返回:
        Tuple[bool, Optional[pt.System]]:
            - bool: 模拟是否成功
            - pt.System: 模拟后的系统对象（失败时为None）
    """
    try:
        # 1. 创建模拟环境副本，避免污染原始状态
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        
        # 2. 构建 System 对象
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        # 3. 设置击球参数
        # 确保 action 包含所有必要参数，使用默认值兜底
        params = {
            'V0': action.get('V0', 1.0),
            'phi': action.get('phi', 0.0),
            'theta': action.get('theta', 0.0),
            'a': action.get('a', 0.0),
            'b': action.get('b', 0.0)
        }
        shot.cue.set_state(**params)
        
        # 4. 执行模拟 (带超时保护)
        # 注意：这里我们假设 simulate_with_timeout 已经实现（虽然它是 TODO）
        # 如果未实现，我们可以暂时直接用 pt.simulate(shot, inplace=True)
        # 为了健壮性，这里先直接调用 pt.simulate，后续如果实现了 simulate_with_timeout 再替换
        # success = simulate_with_timeout(shot)
        
        pt.simulate(shot, inplace=True)
        return True, shot
        
    except Exception as e:
        print(f"[simulate_shot] Simulation failed: {e}")
        return False, None


def simulate_shot_batch(
    action: Dict[str, float],
    balls: Dict[str, Any],
    table: Any,
    n_simulations: int = 10,
    noise_config: Optional[Dict[str, float]] = None
) -> List[Tuple[bool, Optional[pt.System]]]:
    """
    批量模拟击球（带噪声扰动）
    
    参数:
        action: 击球参数字典
        balls: 当前球状态字典
        table: 球桌对象
        n_simulations: 模拟次数
        noise_config: 噪声配置 {'V0': std, 'phi': std, 'theta': std, 'a': std, 'b': std}
    
    返回:
        List[Tuple[bool, pt.System]]: 每次模拟的结果列表
    """
    results = []
    
    for _ in range(n_simulations):
        # 1. 生成带噪声的动作
        noisy_action = add_noise_to_action(action, noise_config)
        
        # 2. 执行单次模拟
        success, shot = simulate_shot(noisy_action, balls, table)
        
        results.append((success, shot))
        
    return results


def add_noise_to_action(
    action: Dict[str, float],
    noise_config: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    为击球动作添加高斯噪声
    
    参数:
        action: 原始击球参数
        noise_config: 噪声标准差配置
    
    返回:
        Dict[str, float]: 添加噪声后的击球参数
    """
    if noise_config is None:
        noise_config = {}
        
    return {
        'V0': np.clip(action.get('V0', 1.0) + np.random.normal(0, noise_config.get('V0', 0)), 0.5, 8.0),
        'phi': (action.get('phi', 0.0) + np.random.normal(0, noise_config.get('phi', 0))) % 360,
        'theta': np.clip(action.get('theta', 0.0) + np.random.normal(0, noise_config.get('theta', 0)), 0, 90),
        'a': np.clip(action.get('a', 0.0) + np.random.normal(0, noise_config.get('a', 0)), -0.5, 0.5),
        'b': np.clip(action.get('b', 0.0) + np.random.normal(0, noise_config.get('b', 0)), -0.5, 0.5)
    }


# ============================================================================
#                               评分层 (Scoring)
# ============================================================================

# -------------------- 评分权重配置 --------------------
SCORE_CONFIG = {
    # 进球奖励
    'pocket_target': 100,        # 目标球进袋
    'pocket_own_extra': 30,      # 额外己方球进袋（非目标球）
    'pocket_opponent': -50,      # 意外打进对方球
    'pocket_eight_legal': 120,   # 合法打进黑8（清台后）
    
    # 犯规惩罚
    'foul_cue_pocket': -150,               # 白球落袋
    'foul_eight_cue_together': -1000,      # 黑8和白球同时落袋（判负）
    'foul_eight_early': -1000,             # 清台前误打黑8（判负）
    'foul_no_contact': -50,                # 空杆（未击中任何球）
    'foul_wrong_first_contact': -50,       # 首碰非目标球
    'foul_no_rail': -50,                   # 无进球且无碰库
    
    # 走位评分配置
    'position_max': 50,         # 走位满分
    'position_weight_distance_to_pocket': 0.3,   # 目标球-袋口距离权重
    'position_weight_cue_to_target': 0.3,        # 母球-目标球距离权重
    'position_weight_angle': 0.4,                # 夹角权重（最重要）
}


def get_ball_position(ball: Any) -> np.ndarray:
    """提取球的位置坐标 (x, y, z)"""
    return ball.state.rvw[0].copy()


def get_pocket_position(pocket: Any) -> np.ndarray:
    """提取袋口的位置坐标"""
    # 袋口位置存储方式可能因版本而异
    if hasattr(pocket, 'center'):
        return np.array([pocket.center[0], pocket.center[1], 0])
    elif hasattr(pocket, 'a'):
        return np.array([pocket.a, pocket.b, 0])
    else:
        # fallback: 尝试直接访问
        return np.array([pocket.x, pocket.y, 0])


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


def score_single_target_position(
    cue_pos: np.ndarray,
    target_ball: Any,
    pocket: Any,
    table: Any
) -> float:
    """
    计算针对单个目标球的走位评分
    
    评分维度:
        1. 目标球到袋口的距离 (越近越好)
        2. 母球到目标球的距离 (适中最好，太近太远都扣分)
        3. 母球-目标球连线 与 目标球-袋口连线 的夹角 (越接近180°越好 = 直球)
    
    参数:
        cue_pos: 母球位置 (x, y, z)
        target_ball: 目标球对象
        pocket: 袋口对象
        table: 球桌对象
    
    返回:
        float: 单目标走位分 (0-50)
    """
    target_pos = get_ball_position(target_ball)
    pocket_pos = get_pocket_position(pocket)
    
    # 只取 xy 平面
    cue_xy = cue_pos[:2]
    target_xy = target_pos[:2]
    pocket_xy = pocket_pos[:2]
    
    # ========== 1. 目标球到袋口距离评分 ==========
    # 距离越近越好，满分15分
    dist_target_to_pocket = np.linalg.norm(target_xy - pocket_xy)
    # 假设球桌对角线长度约 2.5m，距离归一化
    max_dist = 2.5
    dist_ratio = min(dist_target_to_pocket / max_dist, 1.0)
    # 距离越近分越高：0距离=15分，满距离=0分
    dist_pocket_score = 15 * (1 - dist_ratio)
    
    # ========== 2. 母球到目标球距离评分 ==========
    # 适中距离最好（约0.3-0.8m），太近出杆困难，太远精度下降
    dist_cue_to_target = np.linalg.norm(cue_xy - target_xy)
    optimal_min = 0.3   # 最佳距离下限
    optimal_max = 0.8   # 最佳距离上限
    
    if dist_cue_to_target < 0.1:
        # 太近，贴球，严重扣分
        dist_cue_score = 2
    elif dist_cue_to_target < optimal_min:
        # 偏近，部分扣分
        dist_cue_score = 7 + 5 * ((dist_cue_to_target - 0.1) / (optimal_min - 0.1))
    elif dist_cue_to_target <= optimal_max:
        # 最佳距离范围，满分15分
        dist_cue_score = 15
    elif dist_cue_to_target < 1.5:
        # 偏远，逐渐扣分
        dist_cue_score = 15 - 8 * ((dist_cue_to_target - optimal_max) / (1.5 - optimal_max))
    else:
        # 太远
        dist_cue_score = 5
    
    # ========== 3. 夹角评分 ==========
    # 母球-目标球向量
    vec_cue_to_target = target_xy - cue_xy
    # 目标球-袋口向量
    vec_target_to_pocket = pocket_xy - target_xy
    
    # 计算夹角（我们希望接近180度，即直球）
    angle = calculate_angle(vec_cue_to_target, vec_target_to_pocket)
    
    # 夹角评分：180度=20分，90度=10分，0度=0分
    # 线性映射：score = 20 * (angle / 180)
    angle_score = 20 * (angle / 180.0)
    
    # ========== 总分 ==========
    total_score = dist_pocket_score + dist_cue_score + angle_score
    
    return total_score


def score_position(
    cue_pos: np.ndarray,
    remaining_targets: List[str],
    balls_after: Dict[str, Any],
    table: Any
) -> float:
    """
    评估白球停点质量 - 遍历所有剩余目标球，取最高分
    
    对于每个剩余目标球，遍历6个袋口，计算走位评分，
    最终返回所有组合中的最高分
    
    参数:
        cue_pos: 白球最终位置 (x, y, z) 或 (x, y)
        remaining_targets: 剩余己方目标球ID列表
        balls_after: 模拟后的球状态 {ball_id: Ball}
        table: 球桌对象
    
    返回:
        float: 走位评分 (0-100)
    """
    if len(cue_pos) == 2:
        cue_pos = np.array([cue_pos[0], cue_pos[1], 0])
    
    # 如果没有剩余目标球，检查是否该打黑8
    if not remaining_targets:
        remaining_targets = ['8']
    
    best_score = 0.0
    
    for target_id in remaining_targets:
        # 跳过不在场上的球
        if target_id not in balls_after:
            continue
        target_ball = balls_after[target_id]
        # 跳过已经进袋的球 (state.s == 4)
        if target_ball.state.s == 4:
            continue
        
        # 遍历所有袋口
        for pocket_id, pocket in table.pockets.items():
            score = score_single_target_position(cue_pos, target_ball, pocket, table)
            if score > best_score:
                best_score = score
    
    return best_score


def analyze_shot_result(
    shot: pt.System,
    balls_state_before: Dict[str, int],
    target_id: str,
    my_targets: List[str]
) -> Dict[str, Any]:
    """
    分析单次模拟的击球结果
    
    参数:
        shot: 模拟后的 System 对象
        balls_state_before: 模拟前每个球的 state.s 字典 {ball_id: int}
        target_id: 本次瞄准的目标球ID
        my_targets: 己方所有目标球列表
    
    返回:
        Dict: {
            'success': bool,           # 目标球是否进袋（且无致命犯规）
            'foul': bool,              # 是否犯规
            'foul_type': str,          # 犯规类型描述
            'foul_penalty': float,     # 犯规惩罚分
            'own_pocketed': List[str], # 进袋的己方球
            'opp_pocketed': List[str], # 进袋的对方球
            'cue_pocketed': bool,      # 白球是否进袋
            'eight_pocketed': bool,    # 黑8是否进袋
            'cue_final_pos': np.ndarray,   # 白球最终位置
            'first_contact': Optional[str], # 首碰球ID
        }
    """
    # ========== 1. 进袋分析 ==========
    new_pocketed = [
        bid for bid, b in shot.balls.items()
        if b.state.s == 4 and balls_state_before.get(bid, 0) != 4
    ]
    
    cue_pocketed = 'cue' in new_pocketed
    eight_pocketed = '8' in new_pocketed
    target_pocketed = target_id in new_pocketed
    
    # 分类进袋球
    own_pocketed = [bid for bid in new_pocketed if bid in my_targets and bid != 'cue']
    
    # 对方目标球：根据己方目标球推断
    if my_targets and my_targets[0] != '8':
        first_own = my_targets[0]
        if first_own in ['1', '2', '3', '4', '5', '6', '7']:
            opp_ids = ['9', '10', '11', '12', '13', '14', '15']
        else:
            opp_ids = ['1', '2', '3', '4', '5', '6', '7']
    else:
        opp_ids = []
    opp_pocketed = [bid for bid in new_pocketed if bid in opp_ids]
    
    # ========== 2. 事件分析（首碰、碰库）==========
    first_contact = None
    cue_hit_cushion = False
    first_contact_hit_cushion = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        
        # 首碰分析
        if first_contact is None and 'cushion' not in et and 'pocket' not in et and 'cue' in ids:
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact = other_ids[0]
        
        # 碰库分析
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact and first_contact in ids:
                first_contact_hit_cushion = True
    
    # ========== 3. 犯规检测 ==========
    foul = False
    foul_type = None
    foul_penalty = 0
    
    # 判断是否可以打黑8（己方其他球已清空）
    remaining_own = [t for t in my_targets if t != '8' and balls_state_before.get(t, 0) != 4]
    can_shoot_eight = len(remaining_own) == 0
    
    # === 致命犯规 ===
    if eight_pocketed and cue_pocketed:
        foul, foul_type, foul_penalty = True, 'eight_cue_together', SCORE_CONFIG['foul_eight_cue_together']
    elif eight_pocketed and not can_shoot_eight:
        foul, foul_type, foul_penalty = True, 'eight_early', SCORE_CONFIG['foul_eight_early']
    elif target_id == '8' and cue_pocketed:
        foul, foul_type, foul_penalty = True, 'eight_cue_pocket', SCORE_CONFIG['foul_eight_cue_together']
    # === 严重犯规 ===
    elif cue_pocketed:
        foul, foul_type, foul_penalty = True, 'cue_pocket', SCORE_CONFIG['foul_cue_pocket']
    elif eight_pocketed and target_id != '8':
        foul, foul_type, foul_penalty = True, 'eight_accident', SCORE_CONFIG['foul_eight_early']
    # === 普通犯规 ===
    elif first_contact is None:
        foul, foul_type, foul_penalty = True, 'no_contact', SCORE_CONFIG['foul_no_contact']
    elif first_contact not in my_targets:
        foul, foul_type, foul_penalty = True, 'wrong_first_contact', SCORE_CONFIG['foul_wrong_first_contact']
    elif not new_pocketed and not cue_hit_cushion and not first_contact_hit_cushion:
        foul, foul_type, foul_penalty = True, 'no_rail', SCORE_CONFIG['foul_no_rail']
    
    # ========== 4. 成功判定 ==========
    success = target_pocketed and not foul
    
    # ========== 5. 白球最终位置 ==========
    cue_final_pos = shot.balls['cue'].state.rvw[0].copy()
    
    return {
        'success': success,
        'foul': foul,
        'foul_type': foul_type,
        'foul_penalty': foul_penalty,
        'own_pocketed': own_pocketed,
        'opp_pocketed': opp_pocketed,
        'cue_pocketed': cue_pocketed,
        'eight_pocketed': eight_pocketed,
        'cue_final_pos': cue_final_pos,
        'first_contact': first_contact,
    }


def score_shot(
    shot: pt.System,
    balls_state_before: Dict[str, int],
    target_id: str,
    my_targets: List[str],
    table: Any
) -> Dict[str, Any]:
    """
    单杆综合评分 = 进球奖励 + 犯规惩罚 + 走位评分
    
    参数:
        shot: 模拟后的 System 对象
        balls_state_before: 模拟前每个球的 state.s 字典
        target_id: 本次瞄准的目标球
        my_targets: 己方目标球列表
        table: 球桌对象
    
    返回:
        Dict: {
            'total_score': float,      # 综合得分
            'pocket_score': float,     # 进球得分
            'foul_penalty': float,     # 犯规惩罚
            'position_score': float,   # 走位得分
            'result': Dict,            # 详细结果 (analyze_shot_result 的输出)
        }
    """
    # 1. 分析击球结果
    result = analyze_shot_result(shot, balls_state_before, target_id, my_targets)
    
    # 2. 进球奖励
    pocket_score = 0.0
    if result['success']:
        pocket_score += SCORE_CONFIG['pocket_target']
    
    # 额外己方球进袋奖励
    extra_own = [bid for bid in result['own_pocketed'] if bid != target_id]
    pocket_score += len(extra_own) * SCORE_CONFIG['pocket_own_extra']
    
    # 意外打进对方球惩罚
    pocket_score += len(result['opp_pocketed']) * SCORE_CONFIG['pocket_opponent']
    
    # 合法打进黑8奖励
    remaining_own = [t for t in my_targets if t != '8' and balls_state_before.get(t, 0) != 4]
    if result['eight_pocketed'] and target_id == '8' and len(remaining_own) == 0 and not result['cue_pocketed']:
        pocket_score += SCORE_CONFIG['pocket_eight_legal']
    
    # 3. 犯规惩罚
    foul_penalty = result['foul_penalty'] if result['foul'] else 0.0
    
    # 4. 走位评分（只有成功且无致命犯规时才计算）
    position_score = 0.0
    if result['success'] and foul_penalty > -500:
        # 计算剩余目标球
        remaining_after = [
            t for t in my_targets 
            if t not in result['own_pocketed'] and t != target_id and t != '8'
        ]
        # 如果己方球清空，下一杆打黑8
        if not remaining_after:
            remaining_after = ['8']
        
        position_score = score_position(
            result['cue_final_pos'],
            remaining_after,
            shot.balls,
            table
        )
    
    # 5. 综合得分
    total_score = pocket_score + foul_penalty + position_score
    
    return {
        'total_score': total_score,
        'pocket_score': pocket_score,
        'foul_penalty': foul_penalty,
        'position_score': position_score,
        'result': result,
    }


# ============================================================================
#                              求解器层 (Solver)
# ============================================================================

def solve_direct_shot(
    cue_ball: Any,
    target_ball: Any,
    pocket: Any,
    balls: Dict[str, Any],
    table: Any
) -> List[Dict[str, Any]]:
    """
    Direct 方案几何求解器
    
    计算白球直接击打目标球入袋的击球参数
    
    参数:
        cue_ball: 白球对象
        target_ball: 目标球对象
        pocket: 目标袋口
        balls: 所有球的状态
        table: 球桌对象
    
    返回:
        List[Dict]: 可行解列表，每个元素包含:
            {
                'phi': float,           # 击球角度
                'V0': float,            # 建议力度
                'target_id': str,       # 目标球ID
                'pocket_id': str,       # 袋口ID
                'type': 'direct',       # 方案类型
                'difficulty': float,    # 难度评估
                'cut_angle': float,     # 切球角度
                'distance': float,      # 总距离
                'ghost_pos': np.ndarray # 虚拟球位置
            }
    """
    # 1. 准备参数
    R = 0.028575
    if hasattr(cue_ball, 'params') and hasattr(cue_ball.params, 'R'):
        R = cue_ball.params.R
        
    cue_pos = get_ball_position(cue_ball)
    target_pos = get_ball_position(target_ball)
    pocket_pos = get_pocket_position(pocket)
    
    # 2. 辅助函数：障碍检测
    def is_segment_clear(p1: np.ndarray, p2: np.ndarray, exclude_ids: List[str]) -> bool:
        """检测线段 p1-p2 是否被阻挡"""
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-6:
            return True
        
        u = vec / length
        
        for bid, ball in balls.items():
            # 跳过排除的球（白球、目标球）和已进袋的球
            if bid in exclude_ids:
                continue
            if ball.state.s == 4:
                continue
            
            b_pos = get_ball_position(ball)
            # 计算球心到线段的距离
            # 投影点 t = (b_pos - p1) . u
            t = np.dot(b_pos - p1, u)
            
            if t < 0:
                # 投影在线段起点之前，检查到起点的距离
                dist = np.linalg.norm(b_pos - p1)
            elif t > length:
                # 投影在线段终点之后，检查到终点的距离
                dist = np.linalg.norm(b_pos - p2)
            else:
                # 投影在线段上，计算垂直距离
                closest = p1 + t * u
                dist = np.linalg.norm(b_pos - closest)
            
            # 判定阈值：2*R (球与球相切)
            # 稍微留一点余量，防止浮点误差导致误判
            if dist < 2 * R - 1e-5:
                return False
        
        return True

    # 3. 计算 Target -> Pocket (TP) 路径
    vec_tp = pocket_pos - target_pos
    vec_tp[2] = 0  # 忽略高度
    dist_tp = np.linalg.norm(vec_tp)
    
    if dist_tp < 1e-6:
        return [] # 已经在袋口
        
    u_tp = vec_tp / dist_tp
    
    # 检查 TP 路径是否有障碍 (排除 Target)
    # 注意：Pocket 是一个点，实际上袋口有宽度，这里简化为点对点
    if not is_segment_clear(target_pos, pocket_pos, exclude_ids=[target_ball.id, cue_ball.id]):
        return []

    # 4. 计算 Ghost Ball 位置
    # Ghost Ball 是白球击打目标球瞬间白球球心的位置
    # 它位于目标球沿进球方向反向 2R 处
    ghost_pos = target_pos - u_tp * (2 * R)
    
    # 5. 计算 Cue -> Ghost (CG) 路径
    vec_cg = ghost_pos - cue_pos
    vec_cg[2] = 0
    dist_cg = np.linalg.norm(vec_cg)
    
    if dist_cg < 1e-6:
        # 白球就在 Ghost Ball 位置（贴球），需要特殊处理
        # 这里简单认为可以直接打
        u_cg = u_tp 
    else:
        u_cg = vec_cg / dist_cg
        
    # 检查 CG 路径是否有障碍 (排除 Cue, Target)
    if not is_segment_clear(cue_pos, ghost_pos, exclude_ids=[cue_ball.id, target_ball.id]):
        return []
        
    # 6. 计算切角 (Cut Angle)
    # 切角是 白球行进方向(u_cg) 与 目标球行进方向(u_tp) 的夹角
    # cos_theta = dot(u_cg, u_tp)
    dot_product = np.clip(np.dot(u_cg, u_tp), -1.0, 1.0)
    cut_angle_rad = np.arccos(dot_product)
    cut_angle_deg = np.degrees(cut_angle_rad)
    
    # 过滤大切角：超过 80 度极难进球且物理模拟不稳定
    if cut_angle_deg > 80:
        return []
        
    # 7. 计算击球角度 phi
    # atan2(y, x) 返回范围 [-pi, pi]，转换为 [0, 360]
    phi = np.degrees(np.arctan2(u_cg[1], u_cg[0])) % 360
    
    # 8. 估算力度 V0
    # 基础力度 + 距离补偿
    total_dist = dist_cg + dist_tp
    # 经验公式：V0 = 2.0 + 1.5 * dist + 切角补偿
    base_v0 = 2.0 + total_dist * 1.5
    if cut_angle_deg > 45:
        base_v0 *= 1.1 # 大角度稍微加力
    
    suggested_v0 = np.clip(base_v0, 1.5, 7.0)
    
    # 9. 计算难度 (0-100, 越高越难)
    # 距离因子
    diff_dist = min(total_dist / 2.5, 1.0) * 40
    # 切角因子
    diff_angle = (cut_angle_deg / 90.0) * 60
    difficulty = diff_dist + diff_angle
    
    # 10. 构造结果
    solution = {
        'phi': phi,
        'V0': suggested_v0,
        'target_id': target_ball.id,
        'pocket_id': pocket.id if hasattr(pocket, 'id') else 'unknown',
        'type': 'direct',
        'difficulty': difficulty,
        'cut_angle': cut_angle_deg,
        'distance': total_dist,
        'ghost_pos': ghost_pos,
        # 'u_arrival': u_cg,
        # 'u_target_out': u_tp
    }
    
    return [solution]


# ============================================================================
#                             Agent 决策层 (Decision)
# ============================================================================

class NewAgent:
    """
    重构后的 NewAgent - 模块化决策架构
    
    架构:
        1. decision        - 主决策入口
        2. attack_strategy - 进攻策略
        3. defense_strategy - 防守策略
    
    进攻策略流程:
        1. 求解器计算所有 direct 方案
        2. 对所有方案做一次快速模拟+打分
        3. 排除犯规和无法进球的方案
        4. 按评分排序，选出 top-K
        5. 对 top-K 进行精细模拟（10次）
        6. 选出最高分方案
    """

    def __init__(self):
        """初始化 Agent"""
        # 配置参数
        self.top_k = 5                    # 第一阶段筛选数量
        self.fine_simulation_count = 10   # 精细模拟次数
        self.quick_simulation_count = 3   # 快速模拟次数
        
        # 噪声配置
        self.noise_config = {
            'V0': 0.1,
            'phi': 0.5,
            'theta': 0.2,
            'a': 0.02,
            'b': 0.02
        }
        
        # 阈值配置
        self.attack_threshold = 50.0      # 进攻阈值（低于此分数考虑防守）
        
        # 球桌参数（后续根据 table 对象更新）
        self.table_width = 1.12
        self.table_length = 2.24
        self.ball_radius = 0.028575
        
        # 袋口位置（6个袋口）
        # 物理引擎中的实际袋口中心坐标（比几何角点向外偏移）
        # 基于 pooltool 默认参数测量得到
        r_corner = 0.062
        r_side = 0.0645
        
        # 偏移计算：向球桌中心偏移半径的 1/2，提高瞄准稳定性
        offset_c = r_corner * 0.5 * 0.7071  # 45度方向分量 (1/sqrt(2))
        offset_s = r_side * 0.5             # 水平方向分量

        self.pockets = [
            MockPocket(-0.0295 + offset_c, -0.0295 + offset_c, 'lb', r_corner),
            MockPocket(-0.0685 + offset_s, 0.9906,             'lc', r_side),
            MockPocket(-0.0295 + offset_c, 2.0107 - offset_c,  'lt', r_corner),
            MockPocket(1.0201 - offset_c,  -0.0295 + offset_c, 'rb', r_corner),
            MockPocket(1.0591 - offset_s,  0.9906,             'rc', r_side),
            MockPocket(1.0201 - offset_c,  2.0107 - offset_c,  'rt', r_corner)
        ]

    # ========================================================================
    #                              主决策入口
    # ========================================================================

    def decision(
        self,
        balls: Dict[str, Any],
        my_targets: List[str],
        table: Any
    ) -> Dict[str, float]:
        """
        主决策入口
        
        流程:
            1. 检测特殊局面（开球等）
            2. 评估进攻方案
            3. 如果最高分 >= 阈值: 执行进攻
            4. 否则: 评估防守方案，选择最佳防守
        
        参数:
            balls: 当前球状态 {ball_id: Ball}
            my_targets: 己方目标球ID列表
            table: 球桌对象
        
        返回:
            Dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
        """
        # 1. 开球检测
        if self._is_break_shot(balls, table):
            print("[NewAgent] Detected break shot situation.")
            return self._break_shot_action(balls, table, my_targets)
            
        # 2. 进攻评估
        print(f"[NewAgent] Evaluating attack options for targets: {my_targets}")
        best_attack_action, best_attack_score = self.attack_strategy(balls, my_targets, table)
        
        # 3. 决策逻辑
        if best_attack_action is not None and best_attack_score >= self.attack_threshold:
            print(f"[NewAgent] Attack! Score: {best_attack_score:.2f}")
            return best_attack_action
            
        # 4. 防守策略 (暂未实现完全版，使用 Fallback)
        print(f"[NewAgent] No good attack option (Score: {best_attack_score:.2f}). Fallback to random/defense.")
        
        # 简单的防守 Fallback：随机打一杆，或者轻推白球
        # 这里暂时用随机动作替代，直到 defense_strategy 实现
        return self._random_action()

    # ========================================================================
    #                              进攻策略
    # ========================================================================

    def attack_strategy(
        self,
        balls: Dict[str, Any],
        my_targets: List[str],
        table: Any
    ) -> Tuple[Optional[Dict[str, float]], float]:
        """
        进攻策略
        
        流程:
            1. 遍历所有 (目标球, 袋口) 组合
            2. 调用 solve_direct_shot 获取所有 direct 方案
            3. 对所有方案进行快速模拟（3次），排除犯规和无法进球的
            4. 按评分排序，选出 top-K
            5. 对 top-K 进行精细模拟（10次）
            6. 返回最高分方案及其分数
        
        参数:
            balls: 当前球状态
            my_targets: 己方目标球列表
            table: 球桌对象
        
        返回:
            Tuple[Optional[Dict], float]:
                - Dict: 最佳击球动作（无可行方案时为 None）
                - float: 方案评分
        """
        cue_ball = balls['cue']
        pockets = self._get_pockets(table)
        
        # 0. 准备合法目标球 (排除已进袋的球)
        legal_targets = [tid for tid in my_targets if tid in balls and balls[tid].state.s != 4]
        # 如果己方球打完，目标变为黑8
        if not legal_targets:
            legal_targets = ['8']
        
        candidates = []
        
        # 1. 遍历所有组合，收集初步方案
        for target_id in legal_targets:
            # 确保目标球在场上
            if target_id not in balls or balls[target_id].state.s == 4:
                continue
                
            target_ball = balls[target_id]
            
            for pocket in pockets:
                # 2. 调用求解器
                solutions = solve_direct_shot(cue_ball, target_ball, pocket, balls, table)
                
                for sol in solutions:
                    # 构造完整的 action 字典
                    action = {
                        'V0': sol['V0'],
                        'phi': sol['phi'],
                        'theta': 0.0,
                        'a': 0.0,
                        'b': 0.0
                    }
                    candidates.append({
                        'action': action,
                        'target_id': target_id,
                        'pocket_id': sol['pocket_id'],
                        'solution': sol
                    })
        
        if not candidates:
            return None, -1000.0
            
        # 3. 快速筛选
        # 使用 self.quick_simulation_count (默认为3)
        filtered_candidates = self._quick_filter(candidates, balls, my_targets, table)
        
        if not filtered_candidates:
            return None, -1000.0
            
        # 4. 排序并选出 Top-K
        # 排序依据：快速模拟的成功率 > 求解器估算的难度
        filtered_candidates.sort(key=lambda x: (-x.get('quick_success_rate', 0), x['solution']['difficulty']))
        top_candidates = filtered_candidates[:self.top_k]
        
        # 5. 精细评估
        # 使用 self.fine_simulation_count (默认为10)
        final_candidates = self._fine_evaluate(top_candidates, balls, my_targets, table)
        
        if not final_candidates:
            return None, -1000.0
            
        # 6. 选择最佳方案
        # 按 total_score 降序排列
        final_candidates.sort(key=lambda x: -x['fine_result']['total_score'])
        best_option = final_candidates[0]
        
        return best_option['action'], best_option['fine_result']['total_score']

    def _quick_filter(
        self,
        candidates: List[Dict[str, Any]],
        balls: Dict[str, Any],
        my_targets: List[str],
        table: Any
    ) -> List[Dict[str, Any]]:
        """
        快速筛选：对候选方案进行快速模拟，排除不可行方案
        
        策略：
        1. 先进行一次无噪声的确定性模拟
        2. 如果这次模拟犯规或失败，直接剔除
        3. 如果通过，计算一个快速评分（基于结果）
        4. 返回排序后的可行方案列表
        
        参数:
            candidates: 候选方案列表
            balls: 当前球状态
            my_targets: 己方目标球
            table: 球桌对象
        
        返回:
            List[Dict]: 筛选后的可行方案（附带快速评分）
        """
        valid_candidates = []
        
        for cand in candidates:
            # 1. 确定性模拟（不加噪声）
            success, shot = simulate_shot(cand['action'], balls, table)
            
            if not success or shot is None:
                continue
                
            balls_state_before = {bid: ball.state.s for bid, ball in balls.items()}
            
            # 分析结果
            res_analysis = analyze_shot_result(shot, balls_state_before, cand['target_id'], my_targets)
            
            # 剔除犯规或未进球方案
            if res_analysis['foul'] or not res_analysis['success']:
                continue
            
            # 计算快速评分 (复用 score_shot 但只算一次)
            # 注意：score_shot 计算比较全面，包含进球分、犯规分和走位分
            score_res = score_shot(shot, balls_state_before, cand['target_id'], my_targets, table)
            
            cand['quick_success_rate'] = 1.0
            cand['quick_foul_rate'] = 0.0
            cand['quick_score'] = score_res['total_score'] # 记录快速评分
            
            valid_candidates.append(cand)
            
        # 按快速评分降序排序
        valid_candidates.sort(key=lambda x: -x['quick_score'])
            
        return valid_candidates

    def _fine_evaluate(
        self,
        top_candidates: List[Dict[str, Any]],
        balls: Dict[str, Any],
        my_targets: List[str],
        table: Any
    ) -> List[Dict[str, Any]]:
        """
        精细评估：对 top-K 方案进行多次模拟，计算综合评分
        
        参数:
            top_candidates: top-K 候选方案
            balls: 当前球状态
            my_targets: 己方目标球
            table: 球桌对象
        
        返回:
            List[Dict]: 带精细评分的方案列表
        """
        scored_candidates = []
        
        for cand in top_candidates:
            # 批量模拟 (精细次数)
            results = simulate_shot_batch(
                cand['action'], 
                balls, 
                table, 
                n_simulations=self.fine_simulation_count,
                noise_config=self.noise_config
            )
            
            total_score_sum = 0
            success_count = 0
            
            for success, shot in results:
                if not success or shot is None:
                    total_score_sum += -500 # 模拟失败惩罚
                    continue
                
                balls_state_before = {bid: ball.state.s for bid, ball in balls.items()}
                
                # 计算单次综合评分
                score_res = score_shot(shot, balls_state_before, cand['target_id'], my_targets, table)
                total_score_sum += score_res['total_score']
                
                if score_res['result']['success']:
                    success_count += 1
            
            avg_score = total_score_sum / self.fine_simulation_count
            success_rate = success_count / self.fine_simulation_count
            
            cand['fine_result'] = {
                'total_score': avg_score,
                'success_rate': success_rate
            }
            scored_candidates.append(cand)
            
        return scored_candidates

    # ========================================================================
    #                              防守策略
    # ========================================================================

    def defense_strategy(
        self,
        balls: Dict[str, Any],
        my_targets: List[str],
        opp_targets: List[str],
        table: Any
    ) -> Tuple[Optional[Dict[str, float]], float]:
        """
        防守策略
        
        参数:
            balls: 当前球状态
            my_targets: 己方目标球列表
            opp_targets: 对方目标球列表
            table: 球桌对象
        
        返回:
            Tuple[Optional[Dict], float]:
                - Dict: 最佳防守动作（无可行方案时为 None）
                - float: 方案评分
        """
        # TODO: 后续实现防守策略
        raise NotImplementedError

    # ========================================================================
    #                              辅助方法
    # ========================================================================

    def _get_opponent_targets(self, my_targets: List[str]) -> List[str]:
        """获取对手目标球列表"""
        all_solids = [str(i) for i in range(1, 8)]
        all_stripes = [str(i) for i in range(9, 16)]
        
        if my_targets[0] in all_solids or (my_targets == ['8'] and '1' in all_solids):
            return all_stripes
        else:
            return all_solids

    def _is_break_shot(self, balls: Dict[str, Any], table: Any) -> bool:
        """
        检测是否为开球局面
        
        判断条件：
        1. 所有目标球（1-15）都在场上
        2. 目标球的位置分布集中在球堆区域
        """
        # 1. 检查所有目标球是否都在场上
        target_balls = [str(i) for i in range(1, 16)]
        all_on_table = all(
            bid in balls and balls[bid].state.s != 4 
            for bid in target_balls
        )
        if not all_on_table:
            return False
        
        # 2. 检查球堆是否集中（计算位置标准差或平均距离）
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
        return avg_dist < 0.15

    def _break_shot_action(
        self,
        balls: Dict[str, Any],
        table: Any,
        my_targets: List[str]
    ) -> Dict[str, float]:
        """
        生成开球动作
        
        策略：
        1. 找到己方目标球中最靠近发球线（y坐标最小）的球
        2. 确保路径无阻碍（如果有阻碍则换一个球）
        3. 大力击向该球 (7.5 m/s)
        """
        cue_pos = get_ball_position(balls['cue'])
        R = 0.028575
        
        # 1. 筛选合法的瞄准目标（己方球，且排除黑8）
        valid_targets = [tid for tid in my_targets if tid != '8']
        
        # 如果 my_targets 为空或者只剩黑8，退回到所有非黑8球
        if not valid_targets:
            valid_targets = [str(i) for i in range(1, 16) if str(i) != '8']
            
        # 2. 寻找最前端的目标球（y坐标最小），且路径无阻碍
        # 按 y 坐标排序，优先考虑最前面的球
        candidates = []
        for tid in valid_targets:
            if tid in balls and balls[tid].state.s != 4:
                pos = get_ball_position(balls[tid])
                candidates.append({'id': tid, 'pos': pos, 'y': pos[1]})
        
        # 按 y 坐标从小到大排序
        candidates.sort(key=lambda x: x['y'])
        
        best_target_id = None
        target_pos_2d = None
        
        # 简单的障碍检测函数
        def is_path_clear(start_pos, end_pos, ignore_id):
            vec = end_pos - start_pos
            length = np.linalg.norm(vec)
            if length < 1e-6: return True
            u = vec / length
            
            for bid, ball in balls.items():
                if bid == 'cue' or bid == ignore_id or ball.state.s == 4:
                    continue
                b_pos = get_ball_position(ball)[:2]
                
                # 计算球心到路径的距离
                t = np.dot(b_pos - start_pos, u)
                if t < 0 or t > length:
                    continue
                
                closest = start_pos + t * u
                dist = np.linalg.norm(b_pos - closest)
                
                # 判定阈值：2*R (球与球相切)
                if dist < 2 * R - 1e-5:
                    return False
            return True

        # 遍历候选球，找到第一个无阻碍的目标
        for cand in candidates:
            # 检查白球到目标球的路径是否有障碍
            # 只检查 xy 平面
            if is_path_clear(cue_pos[:2], cand['pos'][:2], cand['id']):
                best_target_id = cand['id']
                target_pos_2d = cand['pos'][:2]
                break
        
        # Fallback: 如果所有路径都有阻碍（极少见），或者没找到候选球
        if target_pos_2d is None:
            if candidates:
                # 强行打最前面的球（即使有阻碍）
                target_pos_2d = candidates[0]['pos'][:2]
            else:
                # 极端兜底：瞄准置球点附近
                target_pos_2d = np.array([table.w / 2, table.l * 0.75])
            
        # 3. 计算击球角度
        direction = target_pos_2d - cue_pos[:2]
        phi = np.degrees(np.arctan2(direction[1], direction[0])) % 360
        
        return {
            'V0': 7,  # 大力开球
            'phi': phi,
            'theta': 0.0,
            'a': 0.0,
            'b': 0.0
        }

    def _random_action(self) -> Dict[str, float]:
        """
        生成随机击球动作（fallback）
        
        返回:
            Dict: {
                'V0': [0.5, 8.0] m/s
                'phi': [0, 360] 度
                'theta': [0, 90] 度
                'a', 'b': [-0.5, 0.5]
            }
        """
        return {
            'V0': np.random.uniform(0.5, 8.0),
            'phi': np.random.uniform(0, 360),
            'theta': np.random.uniform(0, 90),
            'a': np.random.uniform(-0.5, 0.5),
            'b': np.random.uniform(-0.5, 0.5)
        }

    def _get_pockets(self, table: Any) -> List[Any]:
        """获取球桌的6个袋口"""
        # 优先使用初始化时定义的优化袋口坐标 (MockPocket)
        if hasattr(self, 'pockets') and self.pockets:
            return self.pockets
        # Fallback: 使用 table 对象中的袋口
        return list(table.pockets.values())

    def _build_system(
        self,
        balls: Dict[str, Any],
        table: Any,
        action: Dict[str, float]
    ) -> pt.System:
        """
        根据当前状态和动作构建 pooltool System 对象
        
        参数:
            balls: 球状态
            table: 球桌
            action: 击球参数
        
        返回:
            pt.System: 可用于模拟的系统对象
        """
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        # 设置参数
        params = {
            'V0': action.get('V0', 1.0),
            'phi': action.get('phi', 0.0),
            'theta': action.get('theta', 0.0),
            'a': action.get('a', 0.0),
            'b': action.get('b', 0.0)
        }
        shot.cue.set_state(**params)
        
        return shot
