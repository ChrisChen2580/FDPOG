# adapted from opacus/privacy_engine.py
# https://github.com/pytorch/opacus/blob/030b723fb89aabf3cde663018bb63e5bb95f197a/opacus/privacy_engine.py
# opacus v1.1.0

from typing import List, Union

from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer
from optimizers.fdp_optimizer import FDP_Optimizer
from torch import optim


class FDP_PrivacyEngine(PrivacyEngine):
    """
     This class defines the customized privacy engine for IS-SGD.
     Specifically, it overwrites the _prepare_optimizer() method from parent class to return DPNSGD_Optimizer
     """

    def _prepare_optimizer(
            self,
            optimizer: optim.Optimizer,
            *,
            noise_multiplier: float,
            max_grad_norm: Union[float, List[float]],
            expected_batch_size: int,
            loss_reduction: str = "mean",
            distributed: bool = False,
            clipping: str = "flat",
            noise_generator=None,
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optimizer = FDP_Optimizer(optimizer=optimizer,
                                 noise_multiplier=noise_multiplier,
                                 max_grad_norm=max_grad_norm,
                                 expected_batch_size=expected_batch_size,
                                 loss_reduction=loss_reduction,
                                 generator=generator,
                                 secure_mode=self.secure_mode)

        return optimizer
