from typing import Any, Mapping, Tuple, Union

import gymnasium

import torch
from torch.distributions import Beta
from skrl.models.torch import GaussianMixin


class BetaMixin(GaussianMixin):
    def __init__(
        self,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
        role: str = "",
    ) -> None:
        """Beta mixin model (stochastic model)

        This mixin mirrors the public interface of ``GaussianMixin`` so that
        it can be used transparently by SKRL algorithms expecting a stochastic
        policy. Internally it parameterizes a Beta distribution per action dim.

        The policy's ``compute`` method must return two tensors for policy role:
        ``alpha_logits`` and ``beta_logits`` (pre-activation). They will be
        transformed via ``softplus`` and shifted by a small epsilon to obtain
        positive concentrations.

        Action scaling:
        - Samples ``y`` are drawn in (0, 1) from Beta(alpha, beta)
        - If ``clip_actions`` is enabled and an action space is present,
          actions are mapped as ``x = y * (high - low) + low`` and clipped.
        - Log-probabilities account for the affine transform with the
          appropriate log-Jacobian correction ``-sum(log(high - low))``.

        Parameters are kept compatible with ``GaussianMixin`` even if not used
        (e.g., log std bounds), to preserve the expected constructor signature.
        """
        # keep the same flags/names as GaussianMixin for compatibility
        self._g_clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)
        if self._g_clip_actions:
            self._g_clip_actions_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            self._g_clip_actions_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

        # Reduction behavior compatible with GaussianMixin
        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._g_reduction = (
            torch.mean
            if reduction == "mean"
            else torch.sum if reduction == "sum" else torch.prod if reduction == "prod" else None
        )

        # Beta-specific state
        self._b_alpha = None
        self._b_beta = None
        self._b_distribution = None
        self._b_num_samples = None
        self._b_min_concentration = 1e-6

    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act stochastically in response to the state of the environment

        The model must provide in ``compute`` for role=="policy":
        ``alpha_logits``, ``beta_logits``, and ``outputs``.
        """
        # map from states/observations to alpha/beta logits
        alpha_logits, beta_logits, outputs = self.compute(inputs, role)

        # convert logits to positive concentrations
        alpha = torch.nn.functional.softplus(alpha_logits) + self._b_min_concentration
        beta = torch.nn.functional.softplus(beta_logits) + self._b_min_concentration

        self._b_alpha = alpha
        self._b_beta = beta
        self._b_num_samples = alpha.shape[0]

        # distribution over (0, 1)
        self._b_distribution = Beta(alpha, beta)

        # sample using reparameterization
        y = self._b_distribution.rsample()

        # map to action space if requested
        if self._g_clip_actions:
            scale = self._g_clip_actions_max - self._g_clip_actions_min
            actions = y * scale + self._g_clip_actions_min
            actions = torch.clamp(actions, min=self._g_clip_actions_min, max=self._g_clip_actions_max)
        else:
            actions = y

        # compute log-probability (with transform correction if needed)
        taken_actions = inputs.get("taken_actions", actions)
        if self._g_clip_actions:
            scale = self._g_clip_actions_max - self._g_clip_actions_min
            # invert affine transform to (0, 1)
            y_taken = (taken_actions - self._g_clip_actions_min) / (scale + 1e-12)
            y_taken = torch.clamp(y_taken, 1e-8, 1.0 - 1e-8)
            log_prob = self._b_distribution.log_prob(y_taken)
            # log |det J| = sum(log(scale)) per action dim
            log_det_jacobian = torch.log(scale + 1e-12)
            log_prob = log_prob - log_det_jacobian
        else:
            y_taken = torch.clamp(taken_actions, 1e-8, 1.0 - 1e-8)
            log_prob = self._b_distribution.log_prob(y_taken)

        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        # mean actions for logging (scaled if needed)
        mean_y = self._b_alpha / (self._b_alpha + self._b_beta)
        if self._g_clip_actions:
            scale = self._g_clip_actions_max - self._g_clip_actions_min
            mean_actions = mean_y * scale + self._g_clip_actions_min
        else:
            mean_actions = mean_y
        outputs["mean_actions"] = mean_actions
        return actions, log_prob, outputs

    def get_entropy(self, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model"""
        if self._b_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._b_distribution.entropy().to(self.device)

    def get_log_std(self, role: str = "") -> torch.Tensor:
        """Return a log-standard-deviation-like tensor for compatibility

        For Beta distributions there is no native log std parameter. This method
        returns the log of the current distribution's standard deviation.
        """
        if self._b_distribution is None:
            return torch.tensor(0.0, device=self.device)
        var = self._b_distribution.variance + 1e-12
        log_std = 0.5 * torch.log(var)
        return log_std

    def distribution(self, role: str = "") -> torch.distributions.Beta:
        """Get the current distribution of the model"""
        return self._b_distribution