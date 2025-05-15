# utils/model_metadata.py
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

class ModelMetadata:
    """模型元数据管理类"""
    
    def __init__(
        self,
        model_name: str,
        timestamp: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        notes: str = ""
    ):
        self.model_name = model_name
        self.timestamp = timestamp or int(time.time())
        self.formatted_time = datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        self.params = params or {}
        self.metrics = metrics or {}
        self.notes = notes
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "formatted_time": self.formatted_time,
            "params": self.params,
            "metrics": self.metrics,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """从字典创建对象"""
        return cls(
            model_name=data["model_name"],
            timestamp=data["timestamp"],
            params=data["params"],
            metrics=data["metrics"],
            notes=data.get("notes", "")
        )
    
    def save(self, filepath: Path) -> None:
        """保存元数据到JSON文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> "ModelMetadata":
        """从JSON文件加载元数据"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


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
    
    def register_model(self, metadata: ModelMetadata, model_path: Path) -> None:
        """注册一个新模型"""
        entry = metadata.to_dict()
        entry["model_path"] = str(model_path)
        self.models.append(entry)
        self._save_registry()
    
    def get_best_model(self, metric_name: str, higher_is_better: bool = True) -> Optional[Dict[str, Any]]:
        """基于指定指标获取最佳模型"""
        if not self.models:
            return None
            
        valid_models = [m for m in self.models if metric_name in m["metrics"]]
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