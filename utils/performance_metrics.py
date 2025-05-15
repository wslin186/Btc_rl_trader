#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
performance_metrics.py —— 交易策略性能评估指标计算
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

def calculate_metrics(equity_curve: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
    """计算策略性能指标"""
    # 确保有索引和值
    if not isinstance(equity_curve, pd.Series):
        equity_curve = pd.Series(equity_curve)
    
    # 计算每日回报率
    returns = equity_curve.pct_change().dropna()
    
    # 基础指标
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    # 风险指标
    daily_std = returns.std()
    annual_vol = daily_std * np.sqrt(252)
    
    # 夏普比率
    risk_free_rate = 0.02 / 252  # 日化无风险利率
    sharpe = (returns.mean() - risk_free_rate) / (returns.std() + 1e-8) * np.sqrt(252)
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    max_drawdown = drawdown.min()
    
    # 卡尔马比率
    calmar = annual_return / (abs(max_drawdown) + 1e-8)
    
    # 胜率计算（如果存在基准）
    win_rate = (returns > 0).sum() / len(returns)
    
    # 基准对比（如果提供基准）
    if benchmark is not None:
        benchmark_returns = benchmark.pct_change().dropna()
        # 对齐两个序列
        common_idx = returns.index.intersection(benchmark_returns.index)
        if len(common_idx) > 0:
            aligned_returns = returns.loc[common_idx]
            aligned_benchmark = benchmark_returns.loc[common_idx]
            
            # 超额收益
            excess_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - (benchmark.iloc[-1] / benchmark.iloc[0])
            
            # 信息比率
            tracking_error = (aligned_returns - aligned_benchmark).std() * np.sqrt(252)
            information_ratio = excess_return / (tracking_error + 1e-8)
        else:
            excess_return = 0
            information_ratio = 0
    else:
        excess_return = 0
        information_ratio = 0
    
    return {
        "总收益率": total_return * 100,  # 百分比
        "年化收益率": annual_return * 100,  # 百分比
        "年化波动率": annual_vol * 100,  # 百分比
        "夏普比率": sharpe,
        "最大回撤": max_drawdown * 100,  # 百分比
        "卡尔马比率": calmar,
        "胜率": win_rate * 100,  # 百分比
        "超额收益": excess_return * 100,  # 百分比
        "信息比率": information_ratio
    }

def analyze_trades(trades: pd.DataFrame) -> Dict[str, float]:
    """分析交易记录"""
    if len(trades) < 2:
        return {}
    
    # 提取买卖交易
    buys = trades[trades["side"] == "BUY"]
    sells = trades[trades["side"] == "SELL"]
    
    # 如果交易数量不匹配，可能有未平仓交易
    min_len = min(len(buys), len(sells))
    
    if min_len == 0:
        return {}
    
    # 计算每笔交易的盈亏
    profits = []
    for i in range(min_len):
        buy_price = buys.iloc[i]["px"]
        sell_price = sells.iloc[i]["px"]
        profit_pct = (sell_price / buy_price - 1) * 100
        profits.append(profit_pct)
    
    profits = np.array(profits)
    
    return {
        "平均交易收益": profits.mean(),
        "交易胜率": (profits > 0).sum() / len(profits) * 100,
        "最大单笔收益": profits.max(),
        "最大单笔亏损": profits.min(),
        "收益亏损比": abs(profits[profits > 0].mean() / (profits[profits < 0].mean() or -1)),
        "交易次数": len(profits)
    }