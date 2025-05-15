#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py —— 多进程 PPO 训练（含 VecNormalize，Windows spawn 安全）
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import platform, multiprocessing
from datetime import datetime
import time
import json
import yaml

import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecMonitor,
    VecNormalize,
)
from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from trading_env import BTCTradingEnv
from utils.data_loader import load_btc_data
from utils.logger import get_logger

logger   = get_logger("train")
CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


# ----------------------------------------------------------------------
def make_env(df, env_cfg: dict, seed: int):
    """工厂函数：为每个子进程构建独立环境实例"""
    def _init():
        env = BTCTradingEnv(df.copy(), **env_cfg)
        env.action_space.seed(seed)
        return env
    return _init
# ----------------------------------------------------------------------

def save_model_metadata(model_path, model, train_df, params, metrics=None):
    """保存模型元数据到JSON文件"""
    timestamp = int(time.time())
    formatted_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    # 基本元数据
    metadata = {
        "model_name": model_path.stem,
        "model_path": str(model_path),
        "timestamp": timestamp,
        "formatted_time": formatted_time,
        "params": params,
        "metrics": metrics or {},
        "data_info": {
            "train_rows": len(train_df),
            "train_start": train_df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
            "train_end": train_df["timestamp"].iloc[-1].strftime("%Y-%m-%d"),
        }
    }
    
    # 保存到JSON文件
    metadata_path = model_path.with_suffix('.json')
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 模型元数据已保存: {metadata_path}")
    return metadata_path


def main() -> None:
    # ---------- 读取配置 ----------
    cfg       = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    data_cfg  = cfg["data"]
    env_cfg   = cfg["env"]
    ppo_cfg   = cfg["ppo"]
    train_cfg = cfg["training"]
    tb_cfg    = cfg["tensorboard"]
    paths_cfg = cfg["paths"]

    # ---------- 加载数据 ----------
    train_df, test_df = load_btc_data(
        Path(data_cfg["processed_path"]),
        train_ratio = data_cfg.get("train_ratio"),
        train_start = data_cfg.get("train_start"),
        train_end   = data_cfg.get("train_end"),
        test_start  = data_cfg.get("test_start"),
        test_end    = data_cfg.get("test_end"),
    )
    logger.info(f"📊 训练集行数: {len(train_df)}")

    # ---------- 构建 VecEnv ----------
    n_envs  = train_cfg["n_envs"]
    n_steps = train_cfg["n_steps"]

    if n_envs == 1:
        vec_env = DummyVecEnv([make_env(train_df, env_cfg, 0)])
    else:
        start_m = "spawn" if platform.system() == "Windows" else "fork"
        vec_env = SubprocVecEnv(
            [make_env(train_df, env_cfg, i) for i in range(n_envs)],
            start_method=start_m,
        )

    vec_env = VecMonitor(vec_env)                      # 回合统计
    vec_env = VecNormalize(                           # 观测 & 奖励归一化
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    logger.info(f"🚀 并行环境: {n_envs} | n_steps / trading_env = {n_steps}")

    # ---------- 创建评估环境 ----------
    eval_env = DummyVecEnv([make_env(test_df.head(min(1000, len(test_df))), env_cfg, 42)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    
    # ---------- TensorBoard 目录 ----------
    tb_dir = None
    if tb_cfg.get("enable", False):
        tb_dir = Path(tb_cfg["log_dir"])
        if not tb_dir.is_absolute():
            tb_dir = Path(__file__).resolve().parents[1] / tb_dir
        tb_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 创建 PPO 模型 ----------
    model = sb3.PPO(
        policy          = ppo_cfg["policy"],
        env             = vec_env,
        n_steps         = n_steps,
        batch_size      = ppo_cfg["batch_size"],
        learning_rate   = ppo_cfg["learning_rate"],
        gamma           = ppo_cfg["gamma"],
        gae_lambda      = ppo_cfg["gae_lambda"],
        ent_coef        = ppo_cfg["ent_coef"],
        vf_coef         = ppo_cfg["vf_coef"],
        clip_range      = ppo_cfg["clip_range"],
        max_grad_norm   = ppo_cfg["max_grad_norm"],
        verbose         = 0,
        tensorboard_log = str(tb_dir) if tb_dir else None,
    )

    # ---------- 训练 ----------
    total_ts = train_cfg["total_timesteps"]
    run_name = datetime.now().strftime("%Y%m%d_%H%M")
    logger.info(f"🏁 开始训练 {total_ts:,} timesteps")
    model.learn(
        total_timesteps = total_ts,
        callback        = [ProgressBarCallback()],
        tb_log_name     = run_name,
    )

    # ---------- 评估模型 ----------
    logger.info("📏 正在评估模型...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    
    # 简单计算胜率（最终资产大于初始资产的比例）
    eval_env.reset()
    n_wins = 0
    n_episodes = 20
    
    for _ in range(n_episodes):
        done = False
        obs = eval_env.reset()
        initial_value = eval_env.venv.envs[0].account_value
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = eval_env.step(action)
            done = dones[0]
            
        final_value = infos[0]["account_value"]
        if final_value > initial_value:
            n_wins += 1
    
    win_rate = n_wins / n_episodes
    logger.info(f"✅ 模型评估 - 平均奖励: {mean_reward:.2f} ± {std_reward:.2f}, 胜率: {win_rate:.1%}")

    # ---------- 保存模型 & VecNormalize ----------
    model_dir = Path(paths_cfg["model_dir"]); model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{paths_cfg['model_name'].replace('.zip','')}_{run_name}.zip"
    model.save(model_path)
    logger.info(f"💾 模型已保存: {model_path}")

    # ---------- 保存元数据 ----------
    metrics = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "win_rate": float(win_rate),
        "train_episodes": n_episodes
    }
    
    params = {
        "learning_rate": float(ppo_cfg["learning_rate"]),
        "gamma": float(ppo_cfg["gamma"]),
        "n_steps": int(n_steps),
        "batch_size": int(ppo_cfg["batch_size"]),
        "ent_coef": float(ppo_cfg["ent_coef"]),
        "total_timesteps": int(total_ts)
    }
    
    metadata_path = save_model_metadata(model_path, model, train_df, params, metrics)
    
    # ---------- 保存 VecNormalize ----------
    norm_path = model_dir / "vec_norm.pkl"
    vec_env.save(norm_path)     # 只保存归一化器统计，不含模型
    logger.info(f"💾 VecNormalize 统计已保存: {norm_path}")

    vec_env.close()
    eval_env.close()

    logger.info(f"🎉 训练完成! 平均奖励: {mean_reward:.2f}, 胜率: {win_rate:.1%}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if platform.system() == "Windows":
        multiprocessing.freeze_support()   # spawn 安全
    main()
