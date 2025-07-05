from typing import Dict, List, Tuple

import torch


class MultiTaskExperienceBuffer:
    def __init__(self, task_names: List[str], total_buffer_size: int, embed_dim: int, device: torch.device):
        self.total_buffer_size = total_buffer_size
        self.embed_dim = embed_dim
        self.device = device
        self.task_names = task_names

        self.q_buffer = torch.zeros((total_buffer_size, embed_dim), dtype=torch.float32, device=device)
        self.action_buffer = torch.zeros(total_buffer_size, dtype=torch.long, device=device)
        self.priority_buffer = torch.zeros(total_buffer_size, dtype=torch.float32, device=device)
        self.task_name_buffer = [""] * total_buffer_size
        
        self.task_indices: Dict[str, List[int]] = {name: [] for name in task_names}
        
        self.ptr = 0
        self.is_full = False

    @property
    def current_size(self) -> int:
        return self.total_buffer_size if self.is_full else self.ptr

    def add(self, q: torch.Tensor, action: torch.Tensor, priority: torch.Tensor, task_name: str):
        batch_size = q.size(0)
        if batch_size == 0:
            return

        for i in range(batch_size):
            # If buffer is full, find the lowest priority sample to replace
            if self.is_full:
                min_prio_idx = int(torch.argmin(self.priority_buffer).item())
                
                # Remove the old sample from its task list
                old_task_name = self.task_name_buffer[min_prio_idx]
                if min_prio_idx in self.task_indices[old_task_name]:
                    self.task_indices[old_task_name].remove(min_prio_idx)
                
                # Use this index for the new sample
                idx_to_replace = min_prio_idx
            else:
                idx_to_replace = self.ptr

            self.q_buffer[idx_to_replace] = q[i].to(self.q_buffer.dtype)
            self.action_buffer[idx_to_replace] = action[i]
            self.priority_buffer[idx_to_replace] = priority[i].to(self.priority_buffer.dtype)
            self.task_name_buffer[idx_to_replace] = task_name
            self.task_indices[task_name].append(idx_to_replace)
            
            if not self.is_full:
                self.ptr += 1
                if self.ptr >= self.total_buffer_size:
                    self.is_full = True
                    self.ptr = 0
    
    def sample(self, total_batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        active_tasks = [name for name, indices in self.task_indices.items() if indices]
        if not active_tasks:
            return torch.empty(0, self.embed_dim, device=self.device), torch.empty(0, dtype=torch.long, device=self.device), {}

        batch_size_per_task = total_batch_size // len(active_tasks)
        
        all_q, all_a = [], []
        all_indices_map: Dict[str, torch.Tensor] = {}

        for task_name in active_tasks:
            task_indices = torch.tensor(self.task_indices[task_name], device=self.device)
            task_priorities = self.priority_buffer[task_indices]
            task_probs = task_priorities / (task_priorities.sum() + 1e-9)
            
            num_to_sample = min(batch_size_per_task, len(task_indices))
            if num_to_sample > 0:
                sampled_rel_indices = torch.multinomial(task_probs, num_to_sample, replacement=True)
                sampled_abs_indices = task_indices[sampled_rel_indices]

                all_q.append(self.q_buffer[sampled_abs_indices])
                all_a.append(self.action_buffer[sampled_abs_indices])
                all_indices_map[task_name] = sampled_abs_indices

        if not all_q:
            return torch.empty(0, self.embed_dim, device=self.device), torch.empty(0, dtype=torch.long, device=self.device), {}
            
        final_q = torch.cat(all_q, dim=0)
        final_a = torch.cat(all_a, dim=0)

        return final_q, final_a, all_indices_map

    def update_priorities(self, indices_by_task: Dict[str, torch.Tensor], td_errors_by_task: Dict[str, torch.Tensor]):
        for task_name, indices in indices_by_task.items():
            if task_name in td_errors_by_task and indices.numel() > 0:
                self.priority_buffer[indices] = td_errors_by_task[task_name].to(self.priority_buffer.dtype)
