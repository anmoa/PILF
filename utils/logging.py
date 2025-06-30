from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, writer: SummaryWriter, global_step: int):
        self.writer = writer
        self.global_step = global_step

    def log_metrics(self, metrics: Dict[str, Any], task_name: str, scope: str = "Train"):
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue

            tag_map = {
                'loss': f'Loss/{scope}/{task_name}',
                'accuracy': f'Accuracy/{scope}/{task_name}',
            }

            if key in tag_map:
                tag = tag_map[key]
            elif key.startswith('gating_'):
                tag = f'Gating/{key.replace("gating_", "").replace("_", " ").title()}/{scope}/{task_name}'
            else:
                tag = f'{key.replace("_", " ").title()}/{scope}/{task_name}'
            
            self.writer.add_scalar(tag, value, self.global_step)
        
        self.writer.flush()
