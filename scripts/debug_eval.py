# debug_eval.py  —— 放在项目根运行：python debug_eval.py
from pathlib import Path, PurePosixPath
import yaml, pandas as pd, numpy as np
from stable_baselines3 import PPO
from env import BTCTradingEnv

ROOT = Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load((ROOT/"config/config.yaml").read_text(encoding="utf-8"))

# ---------- 载入测试集 ----------
proc_path = ROOT / PurePosixPath(cfg["data"]["processed_path"])
df = pd.read_parquet(proc_path)

start = pd.to_datetime(cfg["data"]["test_start"])
end   = pd.to_datetime(cfg["data"]["test_end"])
test_df = df.query("@start <= timestamp <= @end")

# ---------- 载入最新模型 ----------
model_dir  = ROOT / cfg["paths"]["model_dir"]
model_path = max(model_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
model      = PPO.load(model_path)

# ---------- 环境 ----------
env = BTCTradingEnv(test_df, **cfg["env"])
obs, _ = env.reset()
total_reward = 0.0

print(f"Debug  model={model_path.name}  |  test steps = {len(test_df)}")
print("step | action |   reward   |  equity")
print("--------------------------------------")

for step in range(30):                 # 先看 30 步
    action, _ = model.predict(obs, deterministic=True)   # ← 如需随机可改 False
    obs, reward, done, _, info = env.step(action)

    print(f"{step:02d}  |   {int(action)}    | {reward:+.6f} | {info['account_value']:.2f}")
    total_reward += float(reward)
    if done:
        print("Episode ended early")
        break

print("--------------------------------------")
print(f"Accum reward = {total_reward:+.6f}   final equity = {info['account_value']:.2f}")
