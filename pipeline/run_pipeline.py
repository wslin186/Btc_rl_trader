#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_pipeline.py —— 一键：下载 → 预处理 → 训练 [→ 评估]，可选自动启动 TensorBoard

运行指令:
    # 完整流水线 (下载->预处理->训练)
    python pipeline/run_pipeline.py
    
    # 完整流水线 + 评估
    python pipeline/run_pipeline.py --with-eval
    
    # 完整流水线 + 评估 (确定性模式)
    python pipeline/run_pipeline.py --with-eval --det
    
    # 完整流水线 + TensorBoard可视化
    python pipeline/run_pipeline.py --tb
    
    # 跳过某些步骤
    python pipeline/run_pipeline.py --skip-download  # 跳过下载
    python pipeline/run_pipeline.py --skip-prepare   # 跳过预处理
    python pipeline/run_pipeline.py --skip-train     # 跳过训练
    
    # 组合模式
    python pipeline/run_pipeline.py --skip-download --skip-prepare --with-eval --tb
    
    # 仅评估 (跳过前三步)
    python pipeline/run_pipeline.py --skip-download --skip-prepare --skip-train --with-eval
"""

from __future__ import annotations
import argparse
import importlib
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

import yaml

# ---------- 把项目根目录设为 cwd & 加入 sys.path ----------
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                             # 确保相对路径指向项目根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import get_logger        # noqa: E402

logger = get_logger("pipeline")

# ---------------- 动态导入各步骤 ----------------
download_step = importlib.import_module("scripts.download_data").main
prepare_step = importlib.import_module("scripts.prepare_data").main
train_step = importlib.import_module("scripts.train").main
evaluate_step = importlib.import_module("scripts.evaluate_visual").main


# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BTC RL 一键流水线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过 Step1 数据下载")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="跳过 Step2 数据预处理")
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过 Step3 训练")
    parser.add_argument("--with-eval", action="store_true",
                        help="训练后立即评估（Step4）")
    parser.add_argument("--tb", action="store_true",
                        help="训练完成后自动启动 TensorBoard 可视化")
    parser.add_argument("--det", action="store_true",
                        help="评估时使用确定性模式")
    return parser.parse_args()


# ---------------- TensorBoard 启动 ----------------
def launch_tensorboard(cfg: dict) -> None:
    tb_cfg = cfg.get("tensorboard", {})
    if not tb_cfg.get("enable", False):
        logger.warning("TensorBoard 未在 config.yaml 中启用，跳过启动")
        return

    tb_dir = Path(tb_cfg["log_dir"])
    host = tb_cfg.get("host", "127.0.0.1")
    port = tb_cfg.get("port", 6006)

    cmd = [
        "tensorboard",
        f"--logdir={tb_dir}",
        f"--host={host}",
        f"--port={port}",
    ]
    logger.info(f"🎛  启动 TensorBoard: {' '.join(cmd)}")
    subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    webbrowser.open(f"http://{host}:{port}")


# ---------------- 主函数 ----------------
def main() -> None:
    args = parse_args()

    cfg_path = ROOT / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    logger.info(f"📑 使用配置: {cfg_path}")
    logger.info(
        f"   下载区间 : {cfg['download']['start_date']} → {cfg['download']['end_date']}"
    )
    logger.info(
        f"   训练区间 : {cfg['data']['train_start']} → {cfg['data']['train_end']}"
    )
    logger.info(
        f"   测试区间 : {cfg['data']['test_start']}  → {cfg['data']['test_end']}"
    )
    logger.info("-" * 60)

    # --------- Step 1: 下载 ---------
    if args.skip_download:
        logger.info("⏩ 跳过下载")
    else:
        logger.info("🚩 Step 1 / 4  数据下载")
        download_step()

    # --------- Step 2: 预处理 ---------
    if args.skip_prepare:
        logger.info("⏩ 跳过预处理")
    else:
        logger.info("🚩 Step 2 / 4  数据预处理")
        prepare_step()

    # --------- Step 3: 训练 ---------
    did_train = False
    if args.skip_train:
        logger.info("⏩ 跳过训练")
    else:
        logger.info("🚩 Step 3 / 4  PPO 训练")
        train_step()
        did_train = True

    # --------- Step 4: 评估 ---------
    if args.with_eval:
        logger.info("🚩 Step 4 / 4  回测评估")
        evaluate_step(det=args.det)  # 添加det参数
    else:
        logger.info("⏩ 默认跳过评估，可加 --with-eval 执行")

    # --------- TensorBoard 可视化 ---------
    if args.tb and did_train:
        launch_tensorboard(cfg)

    logger.info("✅ Pipeline 执行完毕")


if __name__ == "__main__":
    main()
