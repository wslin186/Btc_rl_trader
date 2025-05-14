#!/usr/bin/trading_env python
# -*- coding: utf-8 -*-
"""
train.py —— 多进程 PPO 训练（含 VecNormalize，Windows spawn 安全）
"""
from __future__ import annotations
import platform, multiprocessing
from pathlib import Path
from datetime import datetime
import yaml

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecMonitor,
    VecNormalize,
)
from stable_baselines3.common.callbacks import ProgressBarCallback

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
    train_df, _ = load_btc_data(
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

    # ---------- 保存模型 & VecNormalize ----------
    model_dir = Path(paths_cfg["model_dir"]); model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{paths_cfg['model_name'].replace('.zip','')}_{run_name}.zip"
    model.save(model_path)
    logger.info(f"💾 模型已保存: {model_path}")

    norm_path = model_dir / "vec_norm.pkl"
    vec_env.save(norm_path)     # 只保存归一化器统计，不含模型
    logger.info(f"💾 VecNormalize 统计已保存: {norm_path}")

    vec_env.close()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if platform.system() == "Windows":
        multiprocessing.freeze_support()   # spawn 安全
    main()
