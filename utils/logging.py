from typing import Any, Dict

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, writer: SummaryWriter, global_step: int):
        self.writer = writer
        self.global_step = global_step

    def log_metrics(self, metrics: Dict[str, Any], task_name: str, scope: str = "Train"):
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue

            tag = None
            if key in ['loss', 'accuracy', 'pi_score', 'surprise', 'tau']:
                tag = f'{key.replace("_", " ").title()}/{scope}/{task_name}'
            elif key.startswith('router_'):
                metric_name = key.replace('router_', '').replace('_', ' ').title()
                tag = f'Router/{metric_name}/{scope}/{task_name}'
            elif key.startswith('vae_kl_loss'):
                metric_name = key.replace('_', ' ').title()
                tag = f'NarrativeGenerator/{metric_name}/{scope}/{task_name}'
            elif key in ['gating_lr_mod', 'expert_lr_mod', 'gating_sigma', 'expert_sigma']:
                metric_name = key.replace('_', ' ').title()
                tag = f'PILR/{metric_name}/{scope}/{task_name}'
            elif key.startswith('gating_'):
                metric_name = key.replace('gating_', '').replace('_', ' ').title()
                tag = f'Gating/{metric_name}/{scope}/{task_name}'
            
            if tag:
                self.writer.add_scalar(tag, value, self.global_step)
            
            self.writer.add_scalar(tag, value, self.global_step)
        
        self.writer.flush()
