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
    # TODO: 实现单次模拟
    raise NotImplementedError


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
    # TODO: 实现批量模拟（带高斯噪声）
    raise NotImplementedError


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
    # TODO: 实现噪声添加
    raise NotImplementedError


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
                ...
            }
    """
    # TODO: 实现 Direct 几何求解
    raise NotImplementedError


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
        self.pockets = []  # TODO: 初始化时填充

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
        # TODO: 实现主决策逻辑
        raise NotImplementedError

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
        # TODO: 实现进攻策略
        raise NotImplementedError

    def _quick_filter(
        self,
        candidates: List[Dict[str, Any]],
        balls: Dict[str, Any],
        my_targets: List[str],
        table: Any
    ) -> List[Dict[str, Any]]:
        """
        快速筛选：对候选方案进行快速模拟，排除不可行方案
        
        参数:
            candidates: 候选方案列表
            balls: 当前球状态
            my_targets: 己方目标球
            table: 球桌对象
        
        返回:
            List[Dict]: 筛选后的可行方案（附带快速评分）
        """
        # TODO: 实现快速筛选
        raise NotImplementedError

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
        # TODO: 实现精细评估
        raise NotImplementedError

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
        # TODO: 实现
        raise NotImplementedError

    def _is_break_shot(self, balls: Dict[str, Any], table: Any) -> bool:
        """检测是否为开球局面"""
        # TODO: 实现
        raise NotImplementedError

    def _break_shot_action(
        self,
        balls: Dict[str, Any],
        table: Any,
        my_targets: List[str]
    ) -> Dict[str, float]:
        """生成开球动作"""
        # TODO: 实现
        raise NotImplementedError

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
        # TODO: 实现
        raise NotImplementedError

    def _get_pockets(self, table: Any) -> List[Any]:
        """获取球桌的6个袋口"""
        # TODO: 实现
        raise NotImplementedError

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
        # TODO: 实现
        raise NotImplementedError
