from typing import Any, Dict

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, writer: SummaryWriter, global_step: int):
        self.writer = writer
        self.global_step = global_step

    def log_metrics(
        self, metrics: Dict[str, Any], task_name: str, scope: str = "Train"
    ):
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or key == "global_step":
                continue

            tag_key = key.replace("_", " ").title()
            category = "Metrics"

            if "loss" in key or "accuracy" in key or "pi_score" in key or "surprise" in key or "tau" in key:
                category = "Core"
            elif key.startswith("gating_"):
                category = "Gating"
                tag_key = tag_key.replace("Gating ", "")
            elif key.startswith("expert_"):
                category = "Expert"
                tag_key = tag_key.replace("Expert ", "")
            elif "lr" in key or "sigma" in key:
                category = "PILR"

            if key.startswith("gating_"):
                tag = f"{scope}/{category}/{tag_key}"
            else:
                tag = f"{scope}/{task_name}/{category}/{tag_key}"
            self.writer.add_scalar(tag, value, self.global_step)

        self.writer.flush()
