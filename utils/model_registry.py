#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_registry.py —— 模型元数据注册表管理
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union


class ModelRegistry:
    """模型注册表，管理多个模型的元数据"""
    
    def __init__(self, registry_dir: Path):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(exist_ok=True)
        self.registry_file = self.registry_dir / "model_registry.json"
        self.models: List[Dict[str, Any]] = []
        self._load_registry()
    
    def _load_registry(self) -> None:
        """加载注册表"""
        if self.registry_file.exists():
            with open(self.registry_file, "r", encoding="utf-8") as f:
                self.models = json.load(f)
        else:
            self.models = []
    
    def _save_registry(self) -> None:
        """保存注册表"""
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.models, f, ensure_ascii=False, indent=2)
    
    def register_model(self, metadata_path: Union[str, Path]) -> None:
        """注册一个新模型"""
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
            
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        # 添加元数据文件路径
        metadata["metadata_path"] = str(metadata_path)
        self.models.append(metadata)
        self._save_registry()
        return metadata
    
    def register_model_from_dict(self, metadata: Dict[str, Any]) -> None:
        """从字典注册模型"""
        self.models.append(metadata)
        self._save_registry()
        
    def get_best_model(self, metric_name: str = "mean_reward", higher_is_better: bool = True) -> Optional[Dict[str, Any]]:
        """基于指定指标获取最佳模型"""
        if not self.models:
            return None
            
        valid_models = [m for m in self.models if metric_name in m.get("metrics", {})]
        if not valid_models:
            return None
            
        if higher_is_better:
            return max(valid_models, key=lambda m: m["metrics"][metric_name])
        else:
            return min(valid_models, key=lambda m: m["metrics"][metric_name])
    
    def get_latest_model(self) -> Optional[Dict[str, Any]]:
        """获取最近保存的模型"""
        if not self.models:
            return None
        return max(self.models, key=lambda m: m["timestamp"])
    
    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有注册的模型"""
        return sorted(self.models, key=lambda m: m["timestamp"], reverse=True)
    
    def filter_models(self, filter_func) -> List[Dict[str, Any]]:
        """使用自定义过滤函数筛选模型"""
        return list(filter(filter_func, self.models))
    
    def delete_model(self, model_path: Union[str, Path]) -> bool:
        """从注册表中删除指定模型"""
        model_path = str(model_path)
        for i, model in enumerate(self.models):
            if model.get("model_path") == model_path:
                del self.models[i]
                self._save_registry()
                return True
        return False
    
    def get_model_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        """根据模型名称查找模型"""
        for model in self.models:
            if model.get("model_name") == model_name:
                return model
        return None
    
    def clear_registry(self) -> None:
        """清空注册表"""
        self.models = []
        self._save_registry()
    
    def get_model_count(self) -> int:
        """获取注册表中的模型数量"""
        return len(self.models)