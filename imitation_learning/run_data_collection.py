import os
import sys
import time
import subprocess
from tqdm import tqdm

def main():
    # 配置
    TOTAL_GAMES = 200000
    NUM_WORKERS = 48  # 这里的进程是完全独立的 python 进程，可以跑满
    GAMES_PER_WORKER = TOTAL_GAMES // NUM_WORKERS
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    WORKER_SCRIPT = os.path.join(BASE_DIR, 'data_collector_worker.py')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    print(f"=== 独立多进程数据采集 ===")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Games per worker: {GAMES_PER_WORKER}")
    
    # 清理旧的 .done 文件
    os.makedirs(DATA_DIR, exist_ok=True)
    for f in os.listdir(DATA_DIR):
        if f.endswith('.done'):
            os.remove(os.path.join(DATA_DIR, f))
            
    processes = []
    
    # 启动子进程
    print("Launching workers...")
    for i in range(NUM_WORKERS):
        cmd = [sys.executable, WORKER_SCRIPT, str(i), str(GAMES_PER_WORKER)]
        # start_new_session=True 让子进程不接受 ctrl+c，需要手动 kill
        # 但我们为了方便还是让它跟随父进程死
        p = subprocess.Popen(cmd)
        processes.append(p)
        
    print(f"Started {len(processes)} processes.")
    
    # 监控进度
    # 通过监控 csv 行数来估算进度比较慢，监控 .done 文件太简单
    # 我们这里只简单显示运行时间，以及检查进程存活状态
    
    start_time = time.time()
    pbar = tqdm(total=NUM_WORKERS, desc="Workers Finished", unit="worker")
    
    finished_count = 0
    
    try:
        while finished_count < NUM_WORKERS:
            # 检查有多少个 worker 生成了 .done 文件
            current_finished = 0
            for i in range(NUM_WORKERS):
                if os.path.exists(os.path.join(DATA_DIR, f'worker_{i}.done')):
                    current_finished += 1
            
            delta = current_finished - finished_count
            if delta > 0:
                pbar.update(delta)
                finished_count = current_finished
            
            # 检查是否有进程意外挂掉
            all_alive = True
            for i, p in enumerate(processes):
                if p.poll() is not None: # 已结束
                    # 检查是否正常结束 (看退出码或 done 文件)
                     pass
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nStopping all workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
            
    end_time = time.time()
    print(f"\nAll Done! Total time: {end_time - start_time:.2f}s")

if __name__ == '__main__':
    main()
