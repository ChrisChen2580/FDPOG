from typing import Optional, Callable

import torch
from opacus.optimizers.optimizer import DPOptimizer, _check_processed_flag, _mark_as_processed, _generate_noise, \
    _get_flat_grad_sample
from torch.optim import Optimizer


class FDP_Optimizer(DPOptimizer):
    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        per_param_norms = [
            g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]

        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

        per_sample_clip_factor = (
                self.max_grad_norm / (per_sample_norms + 1e-6)
        ).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)

            grad_sample = _get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self, max_grad_clip: float):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        Args:
            max_grad_clip: C = max(C_k), for all group k
        """

        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * max_grad_clip,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p.grad)
            _mark_as_processed(p.summed_grad)

    def pre_step(
            self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``
        Args:
            per_sample_clip_bound: Defines the clipping bound for each sample.
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
                :param closure:
        """
        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise(self.max_grad_norm)
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[
        float]:
        if closure is not None:
            with torch.enable_grad():
                closure()
        if self.pre_step():
            return self.original_optimizer.step()
        else:
            return None
