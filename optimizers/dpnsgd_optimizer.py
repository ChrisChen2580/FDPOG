import torch
from opacus.optimizers.optimizer import DPOptimizer, _check_processed_flag, _get_flat_grad_sample, _mark_as_processed


class DPNSGD_Optimizer(DPOptimizer):
    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """
        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        per_sample_clip_factor = self.max_grad_norm / (per_sample_norms + 0.01)

        for p in self.params:
            _check_processed_flag(p.grad_sample)

            grad_sample = _get_flat_grad_sample(p)
            # print(grad_sample.shape)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)
            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad
            _mark_as_processed(p.grad_sample)
