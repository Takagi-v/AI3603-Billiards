'''该脚本用于验证 NewAgent 类的几何求解器 (solve_shot_parameters) 是否正确'''
import numpy as np
import math
from agent import NewAgent

# --- Mock Classes to simulate PoolTool objects ---
class MockParams:
    def __init__(self, R):
        self.R = R

class MockState:
    def __init__(self, pos):
        # pos is [x, y, z]
        self.rvw = np.array([np.array(pos), np.zeros(3), np.zeros(3)])
        self.s = 1 # 1 for rolling/sliding, 4 for pocketed

class MockBall:
    def __init__(self, ball_id, pos, R=0.028575):
        self.id = ball_id
        self.state = MockState(pos)
        self.params = MockParams(R)
    
    def __repr__(self):
        return f"Ball({self.id}, pos={self.state.rvw[0]})"

class MockPocket:
    def __init__(self, center):
        self.center = np.array(center)
    def __repr__(self):
        return f"Pocket(center={self.center})"

# --- Verification Logic ---
def verify_geometric_solver():
    print("========================================")
    print("  开始验证 Geometric Solver (几何求解器)")
    print("========================================")
    
    agent = NewAgent()
    R = 0.028575
    print(f"球半径 R = {R} m")
    
    # Case 1: 直线球 (Straight Shot)
    print("\n[测试 1] 直线球 (Straight Shot)")
    cue = MockBall('cue', [0, 0, 0], R)
    target = MockBall('1', [1.0, 0, 0], R)
    pocket = MockPocket([2.0, 0, 0])
    
    print(f"  白球: {cue.state.rvw[0]}")
    print(f"  目标: {target.state.rvw[0]}")
    print(f"  袋口: {pocket.center}")
    
    params = agent.solve_shot_parameters(cue, target, pocket)
    print(f"  -> 计算结果: {params}")
    
    if params is None:
        print("  FAILED: 应该有解")
    elif abs(params['phi'] - 0) < 0.1 or abs(params['phi'] - 360) < 0.1:
        print("  PASSED: 角度正确 (0度)")
    else:
        print(f"  FAILED: 角度错误, 期望 0, 实际 {params['phi']}")

    # Case 2: 45度切球 (45 Degree Cut)
    print("\n[测试 2] 45度切球")
    # 构造场景：
    # 目标球在 (1, 1), 袋口在 (2, 2) -> 目标球行进方向为 (1, 1) 归一化, 45度方向
    # 白球在 (0, 0)
    # 幻影球位置应该在 目标球 沿 (2,2)-(1,1) 反方向延伸 2R 处
    cue = MockBall('cue', [0, 0, 0], R)
    target = MockBall('1', [1.0, 1.0, 0], R)
    pocket = MockPocket([2.0, 2.0, 0])
    
    params = agent.solve_shot_parameters(cue, target, pocket)
    
    # 手动计算预期值
    u_tp = np.array([1.0, 1.0]) / np.sqrt(2)
    ghost_pos = np.array([1.0, 1.0]) - u_tp * (2 * R)
    u_cg = ghost_pos - np.array([0.0, 0.0])
    expected_phi = np.degrees(np.arctan2(u_cg[1], u_cg[0]))
    
    print(f"  预期 Ghost Ball: {ghost_pos}")
    print(f"  预期 Phi: {expected_phi:.4f}")
    print(f"  -> 计算结果: {params}")
    
    if params and abs(params['phi'] - expected_phi) < 0.1:
        print("  PASSED: 角度符合预期")
    else:
        print("  FAILED: 角度不符合预期")

    # Case 3: 物理不可行 (Impossible Shot)
    print("\n[测试 3] 物理不可行 (角度 > 90)")
    # 目标球 (1, 0), 袋口 (2, 0) -> 需向 +x 击打
    # 白球 (1.5, 0) -> 白球在目标球前方，无法击打
    cue = MockBall('cue', [1.5, 0, 0], R)
    target = MockBall('1', [1.0, 0, 0], R)
    pocket = MockPocket([2.0, 0, 0])
    
    params = agent.solve_shot_parameters(cue, target, pocket)
    print(f"  -> 计算结果: {params}")
    
    if params is None:
        print("  PASSED: 正确识别为无解")
    else:
        print("  FAILED: 应该无解")

def verify_path_checker():
    print("\n========================================")
    print("  开始验证 Path Checker (路径检测器)")
    print("========================================")
    
    agent = NewAgent()
    R = 0.028575
    
    cue = MockBall('cue', [0, 0, 0], R)
    target = MockBall('1', [1.0, 0, 0], R)
    pocket = MockPocket([2.0, 0, 0])
    
    balls = {'cue': cue, '1': target}
    
    # Case 1: 无障碍
    print("\n[测试 1] 无障碍")
    clear = agent.is_path_clear(cue, target, pocket, balls)
    print(f"  Result: {clear}")
    if clear: print("  PASSED")
    else: print("  FAILED")
    
    # Case 2: 障碍在 白球->幻影球 路径上
    print("\n[测试 2] 障碍在 白球->幻影球 路径")
    obs = MockBall('2', [0.5, 0, 0], R) # 正中间挡住
    balls['2'] = obs
    clear = agent.is_path_clear(cue, target, pocket, balls)
    print(f"  障碍位置: {obs.state.rvw[0]}")
    print(f"  Result: {clear}")
    if not clear: print("  PASSED: 正确检测到阻挡")
    else: print("  FAILED: 未检测到阻挡")
    
    # Case 3: 障碍在 目标球->袋口 路径上
    print("\n[测试 3] 障碍在 目标球->袋口 路径")
    obs.state.rvw[0] = np.array([1.5, 0, 0])
    clear = agent.is_path_clear(cue, target, pocket, balls)
    print(f"  障碍位置: {obs.state.rvw[0]}")
    print(f"  Result: {clear}")
    if not clear: print("  PASSED: 正确检测到阻挡")
    else: print("  FAILED: 未检测到阻挡")
    
    # Case 4: 障碍在旁边 (不阻挡)
    print("\n[测试 4] 障碍在旁边 (不阻挡)")
    obs.state.rvw[0] = np.array([0.5, 0.2, 0]) # y=0.2, > 2R (0.057)
    clear = agent.is_path_clear(cue, target, pocket, balls)
    print(f"  障碍位置: {obs.state.rvw[0]}")
    print(f"  Result: {clear}")
    if clear: print("  PASSED: 正确放行")
    else: print("  FAILED: 误报阻挡")

if __name__ == "__main__":
    try:
        verify_geometric_solver()
        verify_path_checker()
        print("\n所有验证完成！")
    except Exception as e:
        print(f"\n验证过程中出错: {e}")
        import traceback
        traceback.print_exc()
