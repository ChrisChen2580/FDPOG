import collections
import os
import time

import numpy as np
import pandas as pd
import torch
from datasets.loaders import get_loader
from functorch import make_functional, vjp, grad
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from utils import *
from tools.DROLoss import LossComputer
from opacus.accountants.rdp import RDPAccountant
from tools.minimize_helper import *


class BaseTrainer:
    """Base class for various training methods"""

    def __init__(self,
                 model,
                 optimizer,
                 train_loader,
                 valid_loader,
                 test_loader,
                 writer,
                 evaluator,
                 device,
                 method="regular",
                 max_epochs=100,
                 num_groups=None,
                 selected_groups=[0, 1],
                 lr=0.01,
                 seed=0
                 ):

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.writer = writer
        self.evaluator = evaluator
        self.device = device

        self.method = method
        self.max_epochs = max_epochs
        self.num_groups = num_groups
        self.num_batch = len(self.train_loader)
        self.selected_groups = selected_groups
        self.epoch = 0
        self.num_layers = get_num_layers(self.model)

        self.lr = lr
        self.seed = seed

    def _train_epoch(self, param_for_step=None):
        criterion = torch.nn.CrossEntropyLoss()
        losses = []
        losses_per_group = np.zeros(self.num_groups)
        all_grad_norms = [[] for _ in range(self.num_groups)]
        group_max_grads = [0] * self.num_groups
        g_B_k_norms = [[] for _ in range(self.num_groups)]

        for _batch_idx, (data, target, group) in enumerate(tqdm(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            losses_per_group = self.get_losses_per_group(criterion, data, target, group, losses_per_group)
            loss.backward()
            per_sample_grads = self.flatten_all_layer_params()

            # get sum of grads over groups over current batch
            if self.method == "regular":
                grad_norms, _, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch_from_vec(
                    per_sample_grads, group)
            elif self.method in ["dpsgd", "dpnsgd", "dpsgd-global-adapt"]:
                grad_norms, _, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch_from_vec(
                    per_sample_grads, group, clipping_bound=self.optimizer.max_grad_norm)
            elif self.method == "dpsgd-f":
                C = self.compute_clipping_bound_per_sample(per_sample_grads, group)
                grad_norms, _, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch_from_vec(
                    per_sample_grads, group)
            elif self.method == "IS":
                grad_norms, _, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch_from_vec(
                    per_sample_grads, group, clipping_bound=self.optimizer.max_grad_norm)


            _, group_counts_batch = split_by_group(data, target, group, self.num_groups, return_counts=1)
            g_B, g_B_k, bar_g_B, bar_g_B_k = self.mean_grads_over(group_counts_batch, sum_grad_vec_batch,
                                                                  sum_clip_grad_vec_batch)


            for i in range(self.num_groups):
                if len(grad_norms[i]) != 0:
                    all_grad_norms[i] = all_grad_norms[i] + grad_norms[i]
                    group_max_grads[i] = max(group_max_grads[i], max(grad_norms[i]))
                    g_B_k_norms[i].append(torch.linalg.norm(g_B_k[i]).item())

            if self.method == "dpsgd-f":
                self.optimizer.step(C)
            elif self.method == "dpsgd-global-adapt":
                next_Z = self._update_Z(per_sample_grads, self.strict_max_grad_norm)
                self.optimizer.step(self.strict_max_grad_norm)
                self.strict_max_grad_norm = next_Z
            else:
                self.optimizer.step()
            losses.append(loss.item())

        group_ave_grad_norms = [np.mean(all_grad_norms[i]) for i in range(self.num_groups)]
        group_norm_grad_ave = [np.mean(g_B_k_norms[i]) for i in range(self.num_groups)]
        if self.method != "regular":
            if self.method in ["dpsgd-f", "dpsgd-global-adapt"]:
                self._update_privacy_accountant()
            if self.method in ["IS"]:
                self.compute_weights(group_ave_grad_norms)

            epsilon = self.privacy_engine.get_epsilon(delta=self.delta)
            print(f"(ε = {epsilon:.2f}, δ = {self.delta})")
            privacy_dict = {"epsilon": epsilon, "delta": self.delta}
            self.writer.record_dict("Privacy", privacy_dict, step=0, save=True, print_results=False)

        return group_ave_grad_norms, group_max_grads, group_norm_grad_ave, \
               losses, losses_per_group / self.num_batch

    def train(self, write_checkpoint=True):
        training_time = 0
        group_loss_epochs = []
        avg_grad_norms_epochs = []
        max_grads_epochs = []
        norm_avg_grad_epochs = []

        while self.epoch < self.max_epochs:
            epoch_start_time = time.time()
            self.model.train()
            avg_grad_norms, max_grads, norm_avg_grad, losses, group_losses, sn = self._train_epoch()

            group_loss_epochs.append([self.epoch, np.mean(losses)] + list(group_losses))
            avg_grad_norms_epochs.append([self.epoch] + list(avg_grad_norms))
            max_grads_epochs.append([self.epoch] + list(max_grads))
            norm_avg_grad_epochs.append([self.epoch] + list(norm_avg_grad))

            epoch_training_time = time.time() - epoch_start_time
            training_time += epoch_training_time

            print(
                f"Train Epoch: {self.epoch} \t"
                f"Loss: {np.mean(losses):.6f} \t"
                f"Loss per group: {group_losses}"
            )

            self._validate()
            self.writer.write_scalar("train/" + "Loss", np.mean(losses), self.epoch)
            self.writer.write_scalars("train/AverageGrad",
                                      {'group' + str(k): v for k, v in enumerate(avg_grad_norms)},
                                      self.epoch)
            self.writer.write_scalars("train/MaxGrad",
                                      {'group' + str(k): v for k, v in enumerate(max_grads)},
                                      self.epoch)
            if write_checkpoint: self.write_checkpoint("latest")
            self.epoch += 1

            if self.epoch == self.max_epochs:
                loss_dict = dict()

                loss_dict["final_loss"] = np.mean(losses)
                loss_dict["final_loss_per_group"] = group_losses
                self.writer.record_dict("final_loss", loss_dict, 0, save=1, print_results=0)

        K = self.num_groups
        # write group_loss to csv
        columns = ["epoch", "train_loss"] + [f"train_loss_{k}" for k in range(K)]
        self.create_csv(group_loss_epochs, columns, "train_loss_per_epochs")

        # write avg_grad_norms to csv
        columns = ["epoch"] + [f"ave_grads_{k}" for k in range(K)]
        self.create_csv(avg_grad_norms_epochs, columns, "avg_grad_norms_per_epochs")

        # write max_grads_epochs to csv
        columns = ["epoch"] + [f"max_grads_{k}" for k in range(K)]
        self.create_csv(max_grads_epochs, columns, "max_grad_norms_per_epochs")

        # write norm_avg_grad to csv
        columns = ["epoch"] + [f"norm_avg_grad_{k}" for k in range(K)]
        self.create_csv(norm_avg_grad_epochs, columns, "norm_avg_grad_per_epochs")

        self.writer.write_scalar("train/" + "avg_train_time_over_epoch",
                                 training_time / (self.max_epochs * 60))  # in minutes
        self._test()

    def create_csv(self, data, columns, title):
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(self.writer.logdir, f"{title}.csv"), index=False)

    def flatten_all_layer_params(self):
        """
        Flatten the parameters of all layers in a model

        Args:
            model: a pytorch model

        Returns:
            a tensor of shape num_samples in a batch * num_params
        """
        per_sample_grad = None
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if per_sample_grad is None:
                    per_sample_grad = torch.flatten(p.grad_sample, 1, -1)
                else:
                    per_sample_grad = torch.cat((per_sample_grad, torch.flatten(p.grad_sample, 1, -1)), 1)
        return per_sample_grad

    def get_per_param_grad(self):
        """
        Flatten the parameters of all layers in a model

        Args:
            model: a pytorch model

        Returns:
            a tensor of shape num_samples in a batch * num_params
        """
        per_param_grad = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                per_param_grad.append(torch.flatten(p.grad_sample, 1, -1))
        return per_param_grad

    def _validate(self):
        valid_results = self.evaluator.validate()
        self.writer.record_dict("Validation", valid_results, self.epoch, save=True)

    def _test(self):
        test_results = self.evaluator.test()
        self.writer.record_dict("Test", test_results, self.epoch, save=True)
        if "accuracy_per_group" in test_results.keys():
            plot_by_group(test_results["accuracy_per_group"], self.writer, data_title="final accuracy_per_group",
                          scale_to_01=1)

    def write_checkpoint(self, tag):
        checkpoint = {
            "epoch": self.epoch,

            "module_state_dict": self.model.state_dict(),
            "opt_state_dict": self.optimizer.state_dict(),
        }

        self.writer.write_checkpoint(f"{tag}", checkpoint)

    def get_losses_per_group(self, criterion, data, target, group, group_losses):
        '''
        Given subset of GroupLabelDataset (data, target, group), computes
        loss of model on each subset (data, target, group=k) and returns
        np array of length num_groups = group_losses + group losses over given data
        '''
        per_group = split_by_group(data, target, group, self.num_groups)
        group_loss_batch = np.zeros(self.num_groups)
        for group_idx, (data_group, target_group) in enumerate(per_group):
            with torch.no_grad():
                if data_group.shape[0] == 0:  # if batch does not contain samples of group i
                    group_loss_batch[group_idx] = 0
                else:
                    group_output = self.model(data_group)
                    group_loss_batch[group_idx] = criterion(group_output, target_group).item()
        group_losses = group_loss_batch + group_losses
        return group_losses

    def get_sum_grad_batch(self, data, targets, groups, criterion, **kwargs):
        data = data.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        per_sample_grads = self.flatten_all_layer_params()

        return self.get_sum_grad_batch_from_vec(per_sample_grads, groups, **kwargs)

    def get_sum_grad_batch_from_vec(self, per_sample_grads, groups, **kwargs):
        if self.method == "dpsgd-f":
            clipping_bounds = self.compute_clipping_bound_per_sample(per_sample_grads, groups)
            grad_norms, clip_grad_norms, sum_grad_vec, sum_clip_grad_vec = get_grad_norms_clip(per_sample_grads, groups,
                                                                                               self.num_groups,
                                                                                               self.clipping_scale_fn,
                                                                                               clipping_bounds=clipping_bounds)
        else:
            grad_norms, clip_grad_norms, sum_grad_vec, sum_clip_grad_vec = get_grad_norms_clip(per_sample_grads, groups,
                                                                                               self.num_groups,
                                                                                               self.clipping_scale_fn,
                                                                                               **kwargs)
        return grad_norms, clip_grad_norms, sum_grad_vec, sum_clip_grad_vec

    def mean_grads_over(self, group_counts, sum_grad_vec, clip_sum_grad_vec):
        g_D = torch.stack(sum_grad_vec, dim=0).sum(dim=0) / sum(group_counts)
        g_D_k = [sum_grad_vec[i] / group_counts[i] for i in range(self.num_groups)]

        bar_g_D = torch.stack(clip_sum_grad_vec, dim=0).sum(dim=0) / sum(group_counts)
        bar_g_D_k = [clip_sum_grad_vec[i] / group_counts[i] for i in range(self.num_groups)]
        return g_D, g_D_k, bar_g_D, bar_g_D_k


class RegularTrainer(BaseTrainer):
    """Class for non-private training"""

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx):
        return 1


class DpsgdTrainer(BaseTrainer):
    """Class for DPSGD training"""

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, clipping_bound):
        return min(1, clipping_bound / grad_norm)

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            sample_rate=0.005,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            **kwargs
        )

        self.privacy_engine = privacy_engine
        self.delta = delta
        self.sample_rate = sample_rate


class DpnsgdTrainer(BaseTrainer):
    """Class for DPSGD training"""

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, clipping_bound):
        return 1 / (grad_norm + 0.01)

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            **kwargs
        )

        self.privacy_engine = privacy_engine
        self.delta = delta


class DpsgdFTrainer(BaseTrainer):
    """Class for DPSGD-F training"""

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, **kwargs):
        clipping_bounds = kwargs["clipping_bounds"]
        return min((clipping_bounds[idx] / grad_norm).item(), 1)

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            base_max_grad_norm=1,  # C0
            counts_noise_multiplier=10,  # noise multiplier applied on mk and ok
            **kwargs
    ):
        """
        Initialization function. Initialize parent class while adding new parameter clipping_bound and noise_scale.

        Args:
            model: model from privacy_engine.make_private()
            optimizer: a DPSGDF_Optimizer
            privacy_engine: DPSGDF_Engine
            train_loader: train_loader from privacy_engine.make_private()
            valid_loader: normal pytorch data loader for validation set
            test_loader: normal pytorch data loader for test set
            writer: writer to tensorboard
            evaluator: evaluate for model performance
            device: device to train the model
            delta: definition in privacy budget
            clipping_bound: C0 in the original paper, defines the threshold of gradients
            counts_noise_multiplier: sigma1 in the original paper, defines noise added to the number of samples with gradient bigger than clipping_bound C0
        """
        super().__init__(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            **kwargs
        )

        self.privacy_engine = privacy_engine
        self.delta = delta
        # new parameters for DPSGDF
        self.base_max_grad_norm = base_max_grad_norm  # C0
        self.counts_noise_multiplier = counts_noise_multiplier  # noise scale applied on mk and ok
        self.sample_rate = 1 / self.num_batch
        self.privacy_step_history = []

    def _update_privacy_accountant(self):
        """
        The Opacus RDP accountant minimizes computation when many SGM steps are taken in a row with the same parameters.
        We alternate between privatizing counts, and gradients with different parameters.
        Accounting is sped up by tracking steps in groups rather than alternating.
        The order of accounting does not affect the privacy guarantee.
        """
        for step in self.privacy_step_history:
            self.privacy_engine.accountant.step(noise_multiplier=step[0], sample_rate=step[1])
        self.privacy_step_history = []

    def compute_clipping_bound_per_sample(self, per_sample_grads, group):
        """compute clipping bound for each sample """
        # calculate mk, ok
        mk = collections.defaultdict(int)
        ok = collections.defaultdict(int)
        # get the l2 norm of gradients of all parameters for each sample, in shape of (batch_size, )
        l2_norm_grad_per_sample = torch.norm(per_sample_grads, p=2, dim=1)  # batch_size

        assert len(group) == len(l2_norm_grad_per_sample)

        for i in range(len(group)):  # looping over batch
            group_idx = group[i].item()
            if l2_norm_grad_per_sample[i].item() > self.base_max_grad_norm:
                mk[group_idx] += 1
            else:
                ok[group_idx] += 1

        # add noise scale to mk and ok
        m2k = {}
        o2k = {}
        m = 0

        # note that some group idx might have 0 sample counts in the batch and we are still adding noise to it
        for group_idx in range(self.num_groups):
            m2k[group_idx] = mk[group_idx] + torch.normal(0, self.counts_noise_multiplier, (1,)).item()
            m2k[group_idx] = max(int(m2k[group_idx]), 0)
            o2k[group_idx] = ok[group_idx] + torch.normal(0, self.counts_noise_multiplier, (1,)).item()
            o2k[group_idx] = max(int(o2k[group_idx]), 0)
            m += m2k[group_idx]

        # Account for privacy cost of privately estimating group sizes
        # using the built in sampled-gaussian-mechanism accountant.
        # L2 sensitivity of per-group counts vector is always 1,
        # so std use in torch.normal is the same as noise_multiplier in accountant.
        # Accounting is done lazily, see _update_privacy_accountant method.
        self.privacy_step_history.append([self.counts_noise_multiplier, self.sample_rate])
        array = []
        bk = {}
        Ck = {}
        for group_idx in range(self.num_groups):
            bk[group_idx] = m2k[group_idx] + o2k[group_idx]
            # added
            if bk[group_idx] == 0:
                array.append(1)  # when bk = 0, m2k = 0, we have 0/0 = 1
            else:
                array.append(m2k[group_idx] * 1.0 / bk[group_idx])

        for group_idx in range(self.num_groups):
            Ck[group_idx] = self.base_max_grad_norm * (1 + array[group_idx] / (np.mean(array) + 1e-8))

        per_sample_clipping_bound = []
        for i in range(len(group)):  # looping over batch
            group_idx = group[i].item()
            per_sample_clipping_bound.append(Ck[group_idx])

        return torch.Tensor(per_sample_clipping_bound).to(device=self.device)


class DpsgdGlobalAdaptiveTrainer(BaseTrainer):

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, clipping_bound):
        if grad_norm > self.strict_max_grad_norm:
            return min(1, clipping_bound / grad_norm)
        else:
            return clipping_bound / self.strict_max_grad_norm

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            strict_max_grad_norm=100,
            bits_noise_multiplier=10,
            lr_Z=0.01,
            threshold=1.0,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            **kwargs
        )
        self.privacy_engine = privacy_engine
        self.delta = delta
        self.strict_max_grad_norm = strict_max_grad_norm  # Z
        self.bits_noise_multiplier = bits_noise_multiplier
        self.lr_Z = lr_Z
        self.sample_rate = 1 / self.num_batch
        self.privacy_step_history = []
        self.threshold = threshold

    def _update_privacy_accountant(self):
        """
        The Opacus RDP accountant minimizes computation when many SGM steps are taken in a row with the same parameters.
        We alternate between privatizing counts, and gradients with different parameters.
        Accounting is sped up by tracking steps in groups rather than alternating.
        The order of accounting does not affect the privacy guarantee.
        """
        for step in self.privacy_step_history:
            self.privacy_engine.accountant.step(noise_multiplier=step[0], sample_rate=step[1])
        self.privacy_step_history = []

    def _update_Z(self, per_sample_grads, Z):
        # get the l2 norm of gradients of all parameters for each sample, in shape of (batch_size, )
        l2_norm_grad_per_sample = torch.norm(per_sample_grads, p=2, dim=1)
        batch_size = len(l2_norm_grad_per_sample)

        dt = 0  # sample count in a batch exceeding Z * threshold
        for i in range(batch_size):  # looping over batch
            if l2_norm_grad_per_sample[i].item() > self.threshold * Z:
                dt += 1

        dt = dt * 1.0 / batch_size  # percentage of samples in a batch that's bigger than the threshold * Z
        noisy_dt = dt + torch.normal(0, self.bits_noise_multiplier, (1,)).item() * 1.0 / batch_size

        factor = math.exp(- self.lr_Z + noisy_dt)

        next_Z = Z * factor

        self.privacy_step_history.append([self.bits_noise_multiplier, self.sample_rate])
        return next_Z


class FDPTrainer(BaseTrainer):
    """Class for IS training"""

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, clipping_bound):
        return min(1, clipping_bound / grad_norm)

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            sample_rate=0.005,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            **kwargs
        )

        self.privacy_engine = privacy_engine
        self.delta = delta
        # new parameters for IS
        self.sample_rate = sample_rate
        _, group_counts = torch.unique(self.train_loader.dataset.z, return_counts=True)
        self.groups = self.train_loader.dataset.z.cpu()
        self.weight = 1.0  # to update accountant
        self.group_weights = torch.ones(self.num_groups)  # for computing gradient weights
        self.counts_noise_multiplier = 10



