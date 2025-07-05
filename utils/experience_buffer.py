from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class PrototypingExperienceBuffer:
    def __init__(
        self,
        capacity: int,
        embed_dim: int,
        num_experts: int,
        device: torch.device,
        priority_decay: float = 0.99,
    ):
        self.capacity = capacity
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.device = device
        self.priority_decay = priority_decay

        self.q_buffer = torch.zeros((self.capacity, self.embed_dim), dtype=torch.float32, device=device)
        self.action_buffer = torch.zeros((self.capacity, self.num_experts), dtype=torch.float32, device=device)
        self.priority_buffer = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        
        self.prototype_map = torch.full((self.capacity,), -1, dtype=torch.long, device=device)
        self.prototypes = torch.zeros((self.capacity, self.embed_dim), dtype=torch.float32, device=device)
        self.prototype_counts = torch.zeros(self.capacity, dtype=torch.long, device=device)
        self.active_prototypes_mask = torch.zeros(self.capacity, dtype=torch.bool, device=device)

        self.ptr: int = 0
        self.is_full: bool = False

    @property
    def current_size(self) -> int:
        return self.capacity if self.is_full else self.ptr

    @property
    def num_active_prototypes(self) -> int:
        return int(self.active_prototypes_mask.sum())

    def _find_free_prototype_slots(self, num_needed: int) -> torch.Tensor:
        if num_needed == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)
        inactive_indices = torch.where(~self.active_prototypes_mask)[0]
        return inactive_indices[:num_needed]

    @torch.no_grad()
    def add(self, q: torch.Tensor, actions: torch.Tensor, pi_scores: torch.Tensor):
        batch_size = q.size(0)
        if batch_size == 0: return

        # 1. Candidate Selection
        priorities = 1.0 - pi_scores
        mean_pi_score = pi_scores.mean().item()
        replacement_budget = int(batch_size * (1.0 - mean_pi_score))
        if replacement_budget == 0: return

        num_candidates = min(replacement_budget, batch_size)
        cand_indices = torch.topk(priorities, k=num_candidates).indices
        
        q_cands, act_cands, prio_cands = q[cand_indices], actions[cand_indices], priorities[cand_indices]
        q_cands_norm = F.normalize(q_cands, p=2, dim=1)

        # 2. Prototype Matching
        if self.num_active_prototypes > 0:
            active_protos = self.prototypes[self.active_prototypes_mask]
            active_protos_norm = F.normalize(active_protos, p=2, dim=1)
            sims = torch.matmul(q_cands_norm, active_protos_norm.t())
            max_sims, best_indices_in_active = torch.max(sims, dim=1)
            
            active_indices_map = torch.where(self.active_prototypes_mask)[0]
            best_proto_indices = active_indices_map[best_indices_in_active]
            
            needs_new_proto = max_sims < prio_cands
        else:
            needs_new_proto = torch.ones(num_candidates, dtype=torch.bool, device=self.device)
            best_proto_indices = torch.full((num_candidates,), -1, dtype=torch.long, device=self.device)

        # 3. Prototype & Buffer Slot Allocation
        num_new_needed = int(needs_new_proto.sum())
        new_proto_slots = self._find_free_prototype_slots(num_new_needed)
        num_can_create = new_proto_slots.numel()

        if num_can_create < num_new_needed:
            # Not enough slots, must select the highest priority candidates to create prototypes for.
            indices_that_need_new = torch.where(needs_new_proto)[0]
            priorities_of_needing_new = prio_cands[indices_that_need_new]
            
            _, top_priority_indices_to_keep = torch.topk(priorities_of_needing_new, k=num_can_create)
            
            final_indices_to_create_new_proto_for = indices_that_need_new[top_priority_indices_to_keep]

            # Update the needs_new_proto mask to only include the selected candidates.
            needs_new_proto.fill_(False)
            needs_new_proto[final_indices_to_create_new_proto_for] = True

        best_proto_indices[needs_new_proto] = new_proto_slots

        if self.is_full:
            k = min(num_candidates, self.capacity)
            if k == 0: return

            buffer_lowest_priorities, indices_to_replace = torch.topk(self.priority_buffer, k=k, largest=False)
            cand_highest_priorities, cand_indices_to_consider = torch.topk(prio_cands, k=k, largest=True)
            
            insertion_mask = cand_highest_priorities > buffer_lowest_priorities
            
            write_indices = indices_to_replace[insertion_mask]
            
            final_cand_indices_in_k_subset = torch.where(insertion_mask)[0]
            final_cand_indices = cand_indices_to_consider[final_cand_indices_in_k_subset]

            if final_cand_indices.numel() == 0: return
        else:
            num_to_add = min(num_candidates, self.capacity - self.ptr)
            write_indices = torch.arange(self.ptr, self.ptr + num_to_add, device=self.device)
            final_cand_indices = torch.arange(num_to_add, device=self.device)

        # 4. Eviction (vectorized)
        if self.is_full and write_indices.numel() > 0:
            old_proto_ids = self.prototype_map[write_indices]
            valid_mask = old_proto_ids != -1
            if valid_mask.any():
                unique_old_ids, counts = torch.unique(old_proto_ids[valid_mask], return_counts=True)
                self.prototype_counts.scatter_add_(0, unique_old_ids, -counts)
        
        # 5. Writing to Buffers (vectorized)
        self.q_buffer[write_indices] = q_cands[final_cand_indices]
        self.action_buffer[write_indices] = act_cands[final_cand_indices]
        self.priority_buffer[write_indices] = prio_cands[final_cand_indices]
        self.prototype_map[write_indices] = best_proto_indices[final_cand_indices]

        # 6. Prototype Updates (vectorized)
        final_proto_indices = best_proto_indices[final_cand_indices]
        unique_final_protos, counts = torch.unique(final_proto_indices, return_counts=True)
        self.prototype_counts.scatter_add_(0, unique_final_protos, counts)
        self.active_prototypes_mask[unique_final_protos] = True
        
        # This part is tricky to vectorize perfectly without a loop, but let's approximate.
        # A full vectorized update would require complex scatter-add on weighted values.
        # For now, we update the prototype with the mean of new arrivals.
        for proto_id in unique_final_protos:
            is_this_proto = final_proto_indices == proto_id
            q_for_this_proto = q_cands[final_cand_indices][is_this_proto]
            
            old_proto = self.prototypes[proto_id]
            old_count = self.prototype_counts[proto_id] - q_for_this_proto.shape[0]
            
            new_sum_q = q_for_this_proto.sum(dim=0)
            
            self.prototypes[proto_id] = (old_proto * old_count + new_sum_q) / self.prototype_counts[proto_id]

        if not self.is_full:
            self.ptr += write_indices.numel()
            if self.ptr >= self.capacity:
                self.is_full = True
        
        inactive_protos_mask = (self.prototype_counts == 0) & self.active_prototypes_mask
        self.active_prototypes_mask[inactive_protos_mask] = False
        self.prototypes[inactive_protos_mask] = 0.0

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.current_size == 0:
            return torch.empty(0, self.embed_dim, device=self.device), torch.empty(0, self.num_experts, dtype=torch.float32, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)
        
        num_to_sample = min(batch_size, self.current_size)
        buffer_priorities = self.priority_buffer[:self.current_size]
        
        if buffer_priorities.sum() == 0.0:
            indices = torch.randint(0, self.current_size, (num_to_sample,), device=self.device)
        else:
            sampling_probs = buffer_priorities / buffer_priorities.sum()
            indices = torch.multinomial(sampling_probs, num_samples=num_to_sample, replacement=True)
        
        return self.q_buffer[indices], self.action_buffer[indices], indices

    def decay_priorities(self, indices: torch.Tensor):
        if indices.numel() > 0:
            self.priority_buffer[indices] *= self.priority_decay

    def retrieve_similar(self, query_vectors: torch.Tensor, k: int) -> torch.Tensor:
        batch_size = query_vectors.size(0)
        if self.current_size == 0 or k == 0:
            return torch.zeros(batch_size, k, self.embed_dim, device=self.device)
        
        query_normalized = F.normalize(query_vectors, p=2, dim=1)
        buffer_q_normalized = F.normalize(self.q_buffer[:self.current_size], p=2, dim=1)
        sim_to_buffer_q = torch.matmul(query_normalized, buffer_q_normalized.t())

        k_retrieved = min(k, self.current_size)
        _, top_k_indices = torch.topk(sim_to_buffer_q, k=k_retrieved, dim=1)
        
        retrieved_q = self.q_buffer[top_k_indices]
        
        padding_needed = k - k_retrieved
        if padding_needed > 0:
            padding_shape = (batch_size, padding_needed, self.embed_dim)
            padding = torch.zeros(padding_shape, device=self.device)
            retrieved_q = torch.cat([retrieved_q, padding], dim=1)

        return retrieved_q

    def get_cluster_distribution(self) -> Dict[int, int]:
        if self.current_size == 0: return {}
        valid_ids = self.prototype_map[:self.current_size]
        valid_mask = valid_ids != -1
        if not valid_mask.any(): return {}
        unique_clusters, counts = torch.unique(valid_ids[valid_mask], return_counts=True)
        return {int(k): int(v) for k, v in zip(unique_clusters, counts, strict=False)}
