import importlib
import os
from typing import Any, Dict


class Config:
    def __init__(self, model_config_path: str, schedule_path: str):
        model_module = self._import_module(model_config_path)
        schedule_module = self._import_module(schedule_path)

        self.model: Dict[str, Any] = model_module.model_config
        
        if 'mlp_ratio' in self.model and 'embed_dim' in self.model:
            self.model['mlp_dim'] = int(self.model['embed_dim'] * self.model['mlp_ratio'])

        self.train_strategy: Dict[str, Any] = getattr(model_module, 'train_strategy_config', {'strategies': [{'name': 'Standard'}]})
        self.pilr: Dict[str, Any] = getattr(model_module, 'pilr_config', {})
        
        self.schedule: Dict[str, Any] = schedule_module.schedule_config

    def _import_module(self, path: str) -> Any:
        # Convert file system path to Python module path
        # e.g., "configs/a.py" -> "configs.a"
        module_path = os.path.normpath(path)
        module_path = module_path.replace(os.sep, '.')
        if module_path.endswith('.py'):
            module_path = module_path[:-3]
        return importlib.import_module(module_path)

def load_config(model_config_path: str, schedule_path: str) -> Config:
    return Config(model_config_path, schedule_path)
