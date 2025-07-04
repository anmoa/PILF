import argparse
import importlib
import os
from copy import deepcopy
from typing import Any, Dict, Optional


class Config:
    model: Dict[str, Any]
    train_strategy: Dict[str, Any]
    pilr: Optional[Dict[str, Any]]
    pi: Dict[str, Any]
    schedule: Dict[str, Any]

    def __init__(self, args: argparse.Namespace):
        config_module = self._import_module(args.config)
        base_config = config_module.BASE_CONFIG

        self.model = self._build_model_config(base_config, args.router)
        self.train_strategy = self._build_train_strategy(base_config, args.update)
        self.pilr = self._build_pilr_config(base_config, args.lrs)
        self.pi = base_config.get("pi_config", {})

        schedule_module = self._import_module(args.schedule)
        self.schedule = schedule_module.schedule_config
        self.schedule["name"] = os.path.splitext(os.path.basename(args.schedule))[0]

    def _import_module(self, path: str) -> Any:
        module_path = os.path.normpath(path).replace(os.sep, ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]
        return importlib.import_module(module_path)

    def _build_model_config(self, base: Dict, router_key: str) -> Dict:
        config = deepcopy(base["model_config"])
        router_config = base["router_configs"].get(router_key)
        if router_config:
            config.update(router_config)

        if "mlp_ratio" in config and "embed_dim" in config:
            config["mlp_dim"] = int(config["embed_dim"] * config["mlp_ratio"])

        # Ensure model_type is set for create_model factory
        if "router_type" in config:
            config["model_type"] = config["router_type"]

        return config

    def _build_train_strategy(self, base: Dict, update_key: str) -> Dict:
        return base["train_strategy_configs"].get(
            update_key, {"strategies": [{"name": "Standard"}]}
        )

    def _build_pilr_config(self, base: Dict, lrs_key: str) -> Optional[Dict]:
        return base["pilr_configs"].get(lrs_key)


def load_config(args: argparse.Namespace) -> Config:
    return Config(args)
