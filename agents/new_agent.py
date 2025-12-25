import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime

from .agent import Agent


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
        
        # ==================== 多档力度系统 ====================
        # 6档力度：超小力、极小力、小力、中力、大力、极大力
        self.power_levels = {
            'ultra_soft': 1.0,  # 超小力：保守击球/极近距离
            'very_soft': 1.5,   # 极小力：近距离精细控制
            'soft': 2.5,        # 小力：近距离进球
            'medium': 4.0,      # 中力：中距离进球
            'hard': 5.5,        # 大力：远距离进球
            'very_hard': 7.0,   # 极大力：超远距离/穿透
        }
        self.power_names = ['ultra_soft', 'very_soft', 'soft', 'medium', 'hard', 'very_hard']
        
        # ==================== 贝叶斯微调配置 ====================
        self.BAYES_INIT_POINTS = 5   # 初始采样点
        self.BAYES_N_ITER = 8        # 优化迭代次数
        self.BAYES_ENABLE = True     # 是否启用贝叶斯微调
        
        print("[NewAgent] 架构初始化完成（6档力度 + Top-15精细模拟 + 贝叶斯微调）")

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
            print(f"\n  >>> 决策: 大力开球!")
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
        if best_attack and best_attack_score >= attack_threshold:
            action = best_attack['action']
            cushions = best_attack.get('cushions', 0)
            cushion_str = f"{cushions}库" if cushions > 0 else "直球"
            print(f"\n  >>> 决策: 进攻!")
            print(f"      目标: {best_attack['target_id']} -> {best_attack['pocket_id']}")
            print(f"      类型: {best_attack['shot_type']} ({cushion_str})")
            print(f"      力度档位: {best_attack.get('power_level', 'N/A')} (V0={action['V0']:.1f})")
            print(f"      角度: phi={action['phi']:.1f}°")
            print(f"      成功率: {best_attack['success_rate']:.0%}, 走位分: {best_attack['position_score']:.1f}, 总分: {best_attack_score:.1f}")
            print(f"{'='*60}\n")
            return action
        else:
            # 5. 没有高成功率方案，选择次优并进行贝叶斯微调
            print(f"\n  进攻分数不足或无高成功率方案...")
            
            # 尝试从所有方案中选择最佳的（即使成功率不够高）
            if attack_options:
                fallback = attack_options[0]
                action = fallback['action']
                cushions = fallback.get('cushions', 0)
                cushion_str = f"{cushions}库" if cushions > 0 else "直球"
                print(f"\n  >>> 决策: 尝试进攻 (次优方案)")
                print(f"      目标: {fallback['target_id']} -> {fallback['pocket_id']}")
                print(f"      类型: {fallback['shot_type']} ({cushion_str})")
                print(f"      力度档位: {fallback.get('power_level', 'N/A')} (V0={action['V0']:.1f})")
                print(f"      成功率: {fallback['success_rate']:.0%}, 总分: {fallback['final_score']:.1f}")
                
                # === 贝叶斯微调（对次优方案进行优化）===
                if self.BAYES_ENABLE:
                    print(f"  [贝叶斯微调] 对次优方案进行优化...")
                    refined_action, refined_score = self._bayesian_refine(
                        action, balls, legal_targets, table, fallback['target_id']
                    )
                    if refined_score > 30:  # 微调后分数足够高才采用
                        action = refined_action
                        print(f"  [贝叶斯微调] 完成! 分数={refined_score:.1f}")
                        print(f"      V0={action['V0']:.2f}, phi={action['phi']:.1f}°, "
                              f"theta={action['theta']:.1f}°, a={action['a']:.2f}, b={action['b']:.2f}")
                    else:
                        print(f"  [贝叶斯微调] 未找到更好方案 (分数={refined_score:.1f})")
                
                print(f"{'='*60}\n")
                return action
            else:
                # 无任何方案，保守轻打目标球
                print(f"\n  >>> 决策: 保守击球 (极小力轻打目标球)")
                action = self._conservative_shot(balls, legal_targets[0], table)
                print(f"      力度档位: ultra_soft (V0={action['V0']:.1f})")
                print(f"      角度: phi={action['phi']:.1f}°")
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
        
        # 使用力度系统的超小力档位
        return {
            'V0': self.power_levels['ultra_soft'],  # 超小力
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
        生成开球动作（带模拟验证）
        
        策略：
        1. 找到球堆边缘的球（y坐标最小，最靠近白球）
        2. 生成多个开球候选方案（不同力度、角度微调）
        3. 对每个方案进行模拟验证
        4. 选择最安全且最有利的方案（不犯规、不打进黑8、优先进己方球）
        """
        cue_pos = balls['cue'].state.rvw[0]
        
        # 确定己方和对方目标球
        if my_targets and my_targets[0] in [str(i) for i in range(1, 8)]:
            my_ball_ids = [str(i) for i in range(1, 8)]
            opponent_ball_ids = [str(i) for i in range(9, 16)]
        else:
            my_ball_ids = [str(i) for i in range(9, 16)]
            opponent_ball_ids = [str(i) for i in range(1, 8)]
        
        # 只从己方球中找边缘球（y坐标最小的几个，最靠近白球）
        edge_balls = []
        for bid in my_ball_ids:
            if bid not in balls or balls[bid].state.s == 4:
                continue
            pos = balls[bid].state.rvw[0]
            edge_balls.append((bid, pos[1], pos))
        
        # 按 y 坐标排序，取最靠近白球的3个作为候选瞄准点
        edge_balls.sort(key=lambda x: x[1])
        candidate_targets = edge_balls[:3]
        
        # 如果己方边缘球不足3个，补充所有己方球
        if len(candidate_targets) < 3:
            for bid in my_ball_ids:
                if bid not in [t[0] for t in candidate_targets]:
                    if bid in balls and balls[bid].state.s != 4:
                        pos = balls[bid].state.rvw[0]
                        candidate_targets.append((bid, pos[1], pos))
                        if len(candidate_targets) >= 2:  # 只选2个边缘球
                            break
        
        print(f"  [开球] 己方边缘球候选: {[b[0] for b in candidate_targets]}")
        
        # 生成开球候选方案 (精简版: 2目标 × 3角度 × 2力度 = 12个)
        break_candidates = []
        
        for target_id, _, target_pos in candidate_targets[:2]:  # 只取2个目标
            direction = target_pos[:2] - cue_pos[:2]
            base_phi = np.degrees(np.arctan2(direction[1], direction[0])) % 360
            
            # 角度微调范围：-2°, 0, +2°
            for phi_offset in [-2, 0, 2]:
                phi = (base_phi + phi_offset) % 360
                
                # 力度选择
                for v0 in [6.5, 7.5]:
                    action = {'V0': v0, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}
                    break_candidates.append({
                        'action': action,
                        'target_id': target_id,
                        'phi_offset': phi_offset
                    })
        
        print(f"  [开球] 共生成 {len(break_candidates)} 个开球候选方案，开始模拟验证 (每方案3次)...")
        
        # 模拟验证每个方案 (3次模拟)
        NUM_BREAK_SIM = 3
        scored_candidates = []
        
        for candidate in break_candidates:
            action = candidate['action']
            
            total_score = 0
            foul_count = 0
            my_pocketed_total = 0
            first_hit_record = None
            
            for sim_i in range(NUM_BREAK_SIM):
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                # 添加噪声模拟
                noisy_action = self._add_noise(action) if sim_i > 0 else action
                shot.cue.set_state(**noisy_action)
                
                if not simulate_with_timeout(shot, timeout=3):
                    foul_count += 1
                    continue
                
                # 分析结果
                cue_pocketed = shot.balls['cue'].state.s == 4
                eight_pocketed = shot.balls['8'].state.s == 4
                
                my_pocketed = [bid for bid in my_ball_ids 
                              if bid in shot.balls and shot.balls[bid].state.s == 4]
                opponent_pocketed = [bid for bid in opponent_ball_ids 
                                    if bid in shot.balls and shot.balls[bid].state.s == 4]
                
                # 检测首球接触 - 排除 cue_stick 和非球ID
                first_hit = None
                for event in shot.events:
                    if hasattr(event, 'ids') and 'cue' in event.ids:
                        other = [bid for bid in event.ids if bid != 'cue' and bid != 'cue_stick' and bid in balls]
                        if other:
                            first_hit = other[0]
                            break
                
                if sim_i == 0:
                    first_hit_record = first_hit
                
                # 计算单次模拟评分
                sim_score = 0
                is_foul = False
                
                if cue_pocketed:
                    sim_score -= 500
                    is_foul = True
                if eight_pocketed:
                    sim_score -= 1000
                    is_foul = True
                
                if first_hit:
                    if first_hit in opponent_ball_ids:
                        sim_score -= 100
                        is_foul = True
                    elif first_hit in my_ball_ids:
                        sim_score += 50
                
                # 己方进球：递增加分 (1个=80, 2个=160, 3个=240...)
                # 公式: n * 80 * n = 80 * n^2，鼓励多进球
                n_my = len(my_pocketed)
                sim_score += 80 * n_my * n_my
                
                # 对方进球扣分
                sim_score -= len(opponent_pocketed) * 40
                
                if not is_foul:
                    sim_score += 100
                else:
                    foul_count += 1
                
                total_score += sim_score
                my_pocketed_total += len(my_pocketed)
            
            # 汇总评分
            avg_score = total_score / NUM_BREAK_SIM
            foul_rate = foul_count / NUM_BREAK_SIM
            
            # 犯规率高的方案降权
            final_score = avg_score * (1 - foul_rate * 0.5)
            
            candidate['score'] = final_score
            candidate['foul_rate'] = foul_rate
            candidate['avg_my_pocketed'] = my_pocketed_total / NUM_BREAK_SIM
            candidate['first_hit'] = first_hit_record
            
            scored_candidates.append(candidate)
        
        scored_candidates.sort(key=lambda x: -x['score'])
        
        if scored_candidates:
            best = scored_candidates[0]
            action = best['action']
            foul_rate = best.get('foul_rate', 0)
            status = "✓ 安全" if foul_rate < 0.2 else ("⚠ 风险中" if foul_rate < 0.5 else "⛔ 高风险")
            
            print(f"  [开球] 选择: 瞄准={best['target_id']}, V0={action['V0']:.1f}, phi={action['phi']:.1f}°")
            print(f"  [开球] 预测: {status} | 首触={best.get('first_hit', 'N/A')} | "
                  f"犯规率={foul_rate:.0%} | 平均进球={best['avg_my_pocketed']:.1f} | 分数={best['score']:.0f}")
            return action
        else:
            print(f"  [开球] 无有效方案，使用默认开球")
            if candidate_targets:
                target_pos = candidate_targets[0][2]
                direction = target_pos[:2] - cue_pos[:2]
                phi = np.degrees(np.arctan2(direction[1], direction[0])) % 360
            else:
                phi = 90
            return {'V0': 7.0, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}

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
                    
                    # ==================== 多档力度选择 ====================
                    # 根据距离和库数确定合适的力度档位范围
                    # 距离分档: <0.8m 近距离, 0.8-1.5m 中距离, >1.5m 远距离
                    if total_dist < 0.5:
                        # 极近距离：超小力、极小力、小力
                        suitable_powers = ['ultra_soft', 'very_soft', 'soft']
                    elif total_dist < 0.8:
                        # 近距离：极小力、小力、中力
                        suitable_powers = ['very_soft', 'soft', 'medium']
                    elif total_dist < 1.5:
                        # 中距离：小力、中力、大力
                        suitable_powers = ['soft', 'medium', 'hard']
                    else:
                        # 远距离：中力、大力、极大力
                        suitable_powers = ['medium', 'hard', 'very_hard']
                    
                    # 如果有库边，增加一档力度 (最大到 very_hard)
                    if cushions > 0:
                        power_upgrade = {
                            'ultra_soft': 'very_soft',
                            'very_soft': 'soft',
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
            # 无噪声模拟一次
            result = self._single_simulate(
                candidate['action'],
                balls,
                table,
                candidate['target_id'],
                legal_targets
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
        
        print(f"  [阶段2] 精细模拟 Top-{len(top_candidates)} 候选 (15次):")
        
        scored_candidates = []
        for idx, candidate in enumerate(top_candidates):
            # 10次精细模拟
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


    # ==================== 蒙特卡洛模拟 ====================
    
    def _single_simulate(self, action, balls, table, target_id, my_targets):
        """
        无噪声单次模拟
        
        返回:
            dict: {
                'is_success': 是否进球,
                'is_foul': 是否犯规,
                'position_score': 走位评分 (如果进球且无犯规)
            }
        """
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        shot.cue.set_state(**action)
        
        # 保存模拟前状态
        balls_state_before = {bid: balls[bid].state.s for bid in balls}
        
        # 带超时保护的模拟
        if not simulate_with_timeout(shot, timeout=3):
            return {'is_success': False, 'is_foul': True, 'position_score': 0}
        
        # 分析结果
        result = self._analyze_shot_result(shot, balls_state_before, target_id, my_targets)
        
        # 计算走位评分
        position_score = 0
        if result['is_success'] and not result['is_foul']:
            cue_final_pos = result['cue_final_pos']
            remaining = [t for t in my_targets if t != target_id and shot.balls[t].state.s != 4]
            if remaining:
                position_score = self._evaluate_position(cue_final_pos, remaining, shot.balls, table)
        
        result['position_score'] = position_score
        return result
    
    def simulate_and_score(self, action, balls, table, target_id, my_targets, num_simulations=None):
        """
        对单个动作进行蒙特卡洛模拟，返回综合评分
        
        参数:
            num_simulations: 模拟次数，默认使用 self.num_simulation
        
        返回:
            dict: {
                'success_rate': 进球成功率,
                'foul_rate': 犯规率,
                'position_score': 走位平均分,
                'final_score': 综合评分
            }
        """
        if num_simulations is None:
            num_simulations = self.num_simulation
            
        success_count = 0
        foul_count = 0
        position_scores = []
        penalty_total = 0
        
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        
        for i in range(num_simulations):
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
        success_rate = success_count / num_simulations
        foul_rate = foul_count / num_simulations
        avg_position = np.mean(position_scores) if position_scores else 0
        avg_penalty = penalty_total / num_simulations
        
        # 如果存在致命犯规风险（任何一次模拟中出现 -500 以下的惩罚），直接返回极低分
        if penalty_total / num_simulations <= -50:  # 平均惩罚超过50说明有严重犯规
            # 检查是否有致命犯规（-500 或 -1000 的惩罚）
            fatal_penalty = penalty_total <= -500  # 累计惩罚很高说明有多次严重犯规
            if fatal_penalty:
                final_score = -1000  # 直接判死
            else:
                # 普通犯规，正常计算但惩罚权重更高
                final_score = (
                    success_rate * 120  # 进球奖励
                    - foul_rate * 35    # 增加犯规惩罚权重
                    + avg_position * 0.25  # 走位权重
                    + avg_penalty * 0.7   # 增加惩罚权重
                )
        else:
            # 综合评分公式（正常情况）
            final_score = (
                success_rate * 120  # 进球奖励
                - foul_rate * 35    # 犯规惩罚
                + avg_position * 0.3  # 走位权重
                + avg_penalty * 0.2   # 惩罚项
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

    # ==================== 贝叶斯小范围微调 ====================
    
    def _bayesian_refine(self, geo_action, balls, my_targets, table, target_id):
        """
        在几何解附近进行贝叶斯小范围微调
        
        参数:
            geo_action: 几何求解得到的初始动作 {'V0', 'phi', 'theta', 'a', 'b'}
            balls: 球状态
            my_targets: 己方目标球
            table: 球桌
            target_id: 目标球ID
            
        返回:
            (refined_action, score): 微调后的动作和分数
        """
        if not self.BAYES_ENABLE:
            return geo_action, 0
        
        # 定义小范围搜索空间
        pbounds = {
            'V0': (max(0.5, geo_action['V0'] - 1.5), min(8.0, geo_action['V0'] + 1.5)),
            'phi': (geo_action['phi'] - 10, geo_action['phi'] + 10),
            'theta': (0, 12),
            'a': (-0.25, 0.25),
            'b': (-0.25, 0.25)
        }
        
        # 保存击球前状态
        balls_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        
        def reward_fn(V0, phi, theta, a, b):
            """评估单次击球的分数"""
            try:
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                action = {'V0': V0, 'phi': phi % 360, 'theta': theta, 'a': a, 'b': b}
                shot.cue.set_state(**action)
                
                if not simulate_with_timeout(shot, timeout=2):
                    return -100
                
                # 分析结果
                balls_state_before = {bid: balls_snapshot[bid].state.s for bid in balls_snapshot}
                result = self._analyze_shot_result(shot, balls_state_before, target_id, my_targets)
                
                # 计算分数
                score = 0
                if result['is_success']:
                    score += 80
                    # 走位评分
                    pos_score = self._evaluate_position(
                        result['cue_final_pos'], my_targets, shot.balls, table
                    )
                    score += pos_score * 0.5
                
                if result['is_foul']:
                    score += result['penalty']
                
                return score
                
            except Exception as e:
                return -200
        
        try:
            optimizer = BayesianOptimization(
                f=reward_fn,
                pbounds=pbounds,
                random_state=np.random.randint(1000000),
                verbose=0
            )
            
            optimizer.maximize(
                init_points=self.BAYES_INIT_POINTS,
                n_iter=self.BAYES_N_ITER
            )
            
            best = optimizer.max
            params = best['params']
            
            refined_action = {
                'V0': float(params['V0']),
                'phi': float(params['phi'] % 360),
                'theta': float(params['theta']),
                'a': float(params['a']),
                'b': float(params['b']),
            }
            
            return refined_action, best['target']
            
        except Exception as e:
            print(f"  [贝叶斯微调] 失败: {e}")
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
        评估白球停点的质量 (重构版 - 复用通用评分函数)
        
        评估维度:
        1. 基于距离+角度的精细走位评分 (调用 score_position)
        2. 贴库惩罚
        3. 中心区域奖励
        
        返回:
            float: 走位分数 (大约 0-100)
        """
        R = 0.028575
        
        # 确保 cue_pos 是 3D
        if len(cue_pos) == 2:
            cue_pos = np.array([cue_pos[0], cue_pos[1], 0])
        
        # 1. 计算剩余目标球
        remaining = [t for t in my_targets if t in balls_after and balls_after[t].state.s != 4]
        if not remaining:
            remaining = ['8']  # 清台后打黑8
        
        # 2. 使用通用走位评分函数 (0-50 分)
        base_score = score_position(cue_pos, remaining, balls_after, table)
        
        # 3. 贴库惩罚
        dist_to_rail = min(
            cue_pos[0] - R, table.w - R - cue_pos[0],
            cue_pos[1] - R, table.l - R - cue_pos[1]
        )
        rail_penalty = 0
        if dist_to_rail < 0.03:  # 3cm 内算贴库
            rail_penalty = -25
        elif dist_to_rail < 0.08:  # 8cm 内轻微惩罚
            rail_penalty = -10
        
        # 4. 中心区域奖励（便于任意方向出杆）
        center = np.array([table.w / 2, table.l / 2, 0])
        dist_to_center = np.linalg.norm(cue_pos - center)
        max_dist = np.sqrt((table.w / 2) ** 2 + (table.l / 2) ** 2)
        center_bonus = 0
        if dist_to_center < max_dist * 0.3:
            center_bonus = 15
        elif dist_to_center < max_dist * 0.5:
            center_bonus = 5
        
        # 总分 = 基础分 + 贴库惩罚 + 中心奖励
        # 基础分 0-50，贴库惩罚 -25~0，中心奖励 0~15，总分范围约 -25 ~ 65
        # 为了与旧版兼容（0-100范围），加一个偏移
        total_score = base_score + rail_penalty + center_bonus + 25  # 偏移使范围约 0-90
        
        return max(0, total_score)  # 确保非负


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
