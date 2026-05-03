"""Baseline concept bottleneck models."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import RelaxedBernoulli

from models.concept_backbones import build_encoder, build_head
from utils.freezing import freeze_module, unfreeze_module


class PackedARConceptPredictor(nn.Module):
    """Batched per-concept AR predictors for teacher-forced concept training."""

    def __init__(self, n_features: int, num_concepts: int, hidden_dim: int = 50) -> None:
        super().__init__()
        self.n_features = n_features
        self.num_concepts = num_concepts
        self.input_dim = n_features + num_concepts
        self.hidden_dim = hidden_dim

        self.weight1 = nn.Parameter(torch.empty(num_concepts, self.input_dim, hidden_dim))
        self.bias1 = nn.Parameter(torch.empty(num_concepts, hidden_dim))
        self.weight2 = nn.Parameter(torch.empty(num_concepts, hidden_dim, 1))
        self.bias2 = nn.Parameter(torch.empty(num_concepts, 1))
        self.concept_mask: torch.Tensor
        self.register_buffer(
            "concept_mask",
            torch.tril(torch.ones(num_concepts, num_concepts), diagonal=-1),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for concept_idx in range(self.num_concepts):
            nn.init.kaiming_uniform_(self.weight1[concept_idx].transpose(0, 1), a=math.sqrt(5))
            fan_in = self.input_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias1[concept_idx], -bound, bound)
            nn.init.kaiming_uniform_(self.weight2[concept_idx].transpose(0, 1), a=math.sqrt(5))
            bound = 1 / math.sqrt(self.hidden_dim)
            nn.init.uniform_(self.bias2[concept_idx], -bound, bound)

    def forward_train(
        self, intermediate: torch.Tensor, concepts_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed forward pass for training AR CBM"""
        # intermediate: (B, n_features)
        batch_size = intermediate.size(0)
        features = intermediate.unsqueeze(1).expand(
            batch_size, self.num_concepts, self.n_features
        )  # (B, num_concepts, n_features)
        concepts = (
            concepts_true.float().unsqueeze(1).expand(batch_size, self.num_concepts, self.num_concepts)
        )  # (B, num_concepts, num_concepts)
        concepts = concepts * self.concept_mask.unsqueeze(0)
        concept_input = torch.cat(
            [features, concepts], dim=-1
        )  # (B, num_concepts, n_features + num_concepts)

        # view concepts as batch dimension
        concept_input = concept_input.permute(1, 0, 2)  # (num_concepts, B, input_dim)
        hidden = torch.bmm(concept_input, self.weight1) + self.bias1.unsqueeze(1)
        hidden = F.leaky_relu(hidden)
        logits = torch.bmm(hidden, self.weight2) + self.bias2.unsqueeze(1)  # (num_concepts, B, 1)
        probs = torch.sigmoid(logits.squeeze(-1).transpose(0, 1))  # (B, num_concepts)
        hard = torch.bernoulli(probs)
        return probs, hard

    def forward_single(
        self,
        intermediate: torch.Tensor,
        previous_concepts: torch.Tensor | None,
        concept_idx: int,
    ) -> torch.Tensor:
        """Forward pass for a single concept prediction in AR CBM."""
        concept_input = intermediate.new_zeros(intermediate.size(0), self.input_dim)
        concept_input[:, : self.n_features] = intermediate
        if previous_concepts is not None and previous_concepts.size(1) > 0:
            concept_input[:, self.n_features : self.n_features + concept_idx] = previous_concepts

        hidden = concept_input @ self.weight1[concept_idx] + self.bias1[concept_idx]
        hidden = F.leaky_relu(hidden)
        logits = hidden @ self.weight2[concept_idx] + self.bias2[concept_idx]
        return torch.sigmoid(logits)  # (B, 1)


class CBM(nn.Module):
    """
    Model class encompassing all baselines: Hard & Soft Concept Bottleneck Model (CBM),
                                            Concept Embedding Model (CEM), and Autoregressive CBM (AR).

    This class implements the baselines. Depending on the choice of model, only a small part of the full code is used.
    Check the if statements in the forward method to see which part of the code is used for which model.

    Args:
        config (dict): Configuration dictionary containing model and data settings.

    Noteworthy Attributes:
        training_mode (str): The training mode (e.g., "joint", "sequential", "independent").
        concept_learning (str): The concept learning method ("hard", "soft", "embedding", or "autoregressive").
                                This determines the type of method to use
        num_monte_carlo (int): The number of Monte Carlo samples for sampling Gumbel Softmax in AR.
        straight_through (bool): Flag indicating whether to use straight-through gradients.
        curr_temp (float): The current temperature for the Gumbel-Softmax distribution.
    """

    def __init__(self, config) -> None:
        super(CBM, self).__init__()

        # Configuration arguments
        config_model = config.model
        self.num_concepts = config.data.num_concepts
        self.num_classes = config.data.num_classes
        self.encoder_arch = config_model.encoder_arch
        self.head_arch = config_model.head_arch
        self.training_mode = config_model.training_mode
        self.concept_learning = config_model.concept_learning
        if self.concept_learning in ("hard", "autoregressive"):
            self.num_monte_carlo = config_model.num_monte_carlo
            self.straight_through = config_model.straight_through
            self.curr_temp = 1.0
            if self.training_mode == "joint":
                self.num_epochs = config_model.j_epochs
            else:
                self.num_epochs = config_model.t_epochs
        elif self.concept_learning == "embedding":
            self.CEM_embedding = config_model.embedding_size

        # Architectures
        # Encoder h(.)
        self.encoder, n_features, encoder_res = build_encoder(config, config_model)
        if encoder_res is not None:
            self.encoder_res = encoder_res
        if self.concept_learning == "embedding":
            print(
                "Please be aware that our implementation of CEMs is without training on interventions! "
                "This is because we would deem this an unfair comparison to our method that is also not "
                "trained on interventions. Still, be careful when using this CEM code for derivative works"
            )
            self.positive_embeddings = nn.Sequential(
                nn.Linear(n_features, self.num_concepts * self.CEM_embedding, bias=True),
                nn.LeakyReLU(),
            )
            self.negative_embeddings = nn.Sequential(
                nn.Linear(n_features, self.num_concepts * self.CEM_embedding, bias=True),
                nn.LeakyReLU(),
            )
            self.scoring_function = nn.Sequential(
                nn.Linear(self.CEM_embedding * 2, 1, bias=True), nn.Sigmoid()
            )
            self.concept_dim = self.CEM_embedding * self.num_concepts
        else:
            if self.concept_learning == "autoregressive":
                self.concept_predictor = PackedARConceptPredictor(
                    n_features=n_features,
                    num_concepts=self.num_concepts,
                    hidden_dim=50,
                )
            else:  # Hard or Soft CBM
                self.concept_predictor = nn.Linear(n_features, self.num_concepts, bias=True)
            self.concept_dim = self.num_concepts

        # Assume binary concepts
        self.act_c = nn.Sigmoid()

        # Link function g(.)
        if self.num_classes == 2:
            self.pred_dim = 1
        elif self.num_classes > 2:
            self.pred_dim = self.num_classes

        self.head = build_head(self.concept_dim, self.pred_dim, self.head_arch)

    def forward(
        self,
        x: torch.Tensor,
        epoch: int,
        c_true: torch.Tensor | None = None,
        validation: bool = False,
        concepts_train_ar: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through one of the baselines.

        This method performs a forward pass predicting concept probabilities and logits for the target variable.
        It handles different concept learning strategies and training modes, including hard, soft, autoregressive, and embedding-based concepts.

        Args:
            x (torch.Tensor): The input covariates. Shape: (batch_size, input_dims)
            epoch (int): The current epoch number.
            c_true (torch.Tensor, optional): The ground-truth concept values. Required for "independent" training mode. Default is None.
            validation (bool, optional): Flag indicating whether this is a validation pass. Default is False.
            concepts_train_ar (torch.Tensor, optional): Ground-truth concept values for autoregressive training. Default is None.

        Returns:
            tuple: A tuple containing:
                - c_prob (torch.Tensor): Predicted concept probabilities. Shape: (batch_size, num_concepts)
                - y_pred_logits (torch.Tensor): Logits for the target variable. Shape: (batch_size, label_dim)
                - c (torch.Tensor): Predicted hard concept values (if method permits, otherwise the concept representation).
                    Shape: (batch_size, num_concepts, num_monte_carlo) for MCMC sampling or (batch_size, num_concepts) otherwise.
        """

        intermediate = self.encoder(x)
        c_logit = None

        if self.concept_learning == "hard":
            c_prob, c = self._forward_hard(intermediate, epoch, validation)
        elif self.concept_learning == "soft":
            c_prob, c, c_logit = self._forward_soft(intermediate)
        elif self.concept_learning == "autoregressive":
            c_prob, c = self._forward_ar(intermediate, c_true, concepts_train_ar, validation)
        elif self.concept_learning == "embedding":
            c_prob, c = self._forward_cem(intermediate)
        else:
            raise NotImplementedError(f"concept learning method {self.concept_learning} not supported")

        y_pred_logits = self._predict_target(c, c_logit, validation)
        return c_prob, y_pred_logits, c

    def _forward_hard(self, intermediate: torch.Tensor, epoch: int, validation: bool):
        c_logit = self.concept_predictor(intermediate)
        c_prob = self.act_c(c_logit)

        if self.training_mode == "sequential" or validation:
            # Sample from Bernoulli M times, as we don't need to backprop
            c_prob_mcmc = c_prob.unsqueeze(-1).expand(-1, -1, self.num_monte_carlo)
            c = torch.bernoulli(c_prob_mcmc)

        # Relax bernoulli sampling with Gumbel Softmax to allow for backpropagation
        elif self.training_mode == "joint":
            curr_temp = self.compute_temperature(epoch)
            dist = RelaxedBernoulli(temperature=curr_temp, probs=c_prob)
            c_relaxed = dist.rsample([self.num_monte_carlo]).movedim(0, -1)
            if self.straight_through:
                # Straight-Through Gumbel Softmax
                c_hard = (c_relaxed > 0.5) * 1
                c = c_hard - c_relaxed.detach() + c_relaxed
            else:
                # Reparametrization trick.
                c = c_relaxed

        else:
            raise ValueError(f"unsupported training mode {self.training_mode} for hard CBM")

        return c_prob, c

    def _forward_soft(
        self, intermediate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c_logit = self.concept_predictor(intermediate)
        c_prob = self.act_c(c_logit)
        c = torch.empty_like(c_prob)
        return c_prob, c, c_logit

    def _forward_ar(
        self,
        intermediate: torch.Tensor,
        c_true: torch.Tensor | None,
        concepts_train_ar: torch.Tensor | None,
        validation: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if validation:
            return self._forward_ar_validation(intermediate)

        if self.training_mode == "independent":
            # Training the concept encoder
            if c_true is None and concepts_train_ar is not None:
                return self._forward_ar_concepts_training(intermediate, concepts_train_ar)

            # Training the (target) head  with the GT concepts as input
            if c_true is not None and concepts_train_ar is None:
                c_prob = c_true.float()
                c = c_true.float()
                return c_prob, c

            raise ValueError(
                "For independent training of AR, either c_true or concepts_train_ar must be provided, but not both.\n"
                f"{c_true is None = }, {concepts_train_ar is None = }"
            )

        raise NotImplementedError(
            f"Undefined behavior for Autoregressive CBM with {self.training_mode = } and {validation = }."
        )

    def forward_target_from_concepts(
        self, concepts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        concept_probs = concepts.float()
        target_logits = self.head(concept_probs)
        return concept_probs, target_logits

    def _forward_ar_validation(self, intermediate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self.concept_predictor, PackedARConceptPredictor)

        B = intermediate.size(0)
        c_prob = intermediate.new_empty(B, self.num_concepts, self.num_monte_carlo)
        c = intermediate.new_empty(B, self.num_concepts, self.num_monte_carlo)
        expanded_intermediate = intermediate.unsqueeze(1).expand(
            -1, self.num_monte_carlo, -1
        )  # (B, num_monte_carlo, n_features)

        concept = self.concept_predictor.forward_single(intermediate, None, 0)  # (B, 1)
        concept = concept.expand(-1, self.num_monte_carlo)  # (B, num_monte_carlo)
        c_prob[:, 0, :] = concept
        c[:, 0, :] = torch.bernoulli(concept)

        for concept_idx in range(1, self.num_concepts):
            previous_concepts = c[:, :concept_idx, :].permute(
                0, 2, 1
            )  # (B, num_monte_carlo, num_concepts_so_far)
            concept = self.concept_predictor.forward_single(
                expanded_intermediate.reshape(B * self.num_monte_carlo, -1),
                previous_concepts.reshape(B * self.num_monte_carlo, concept_idx),
                concept_idx,
            ).view(B, self.num_monte_carlo)  # (B, num_monte_carlo)
            c_prob[:, concept_idx, :] = concept
            c[:, concept_idx, :] = torch.bernoulli(concept)

        return c_prob, c

    def _forward_ar_concepts_training(
        self, intermediate: torch.Tensor, concepts_train_ar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self.concept_predictor, PackedARConceptPredictor)
        return self.concept_predictor.forward_train(intermediate, concepts_train_ar)

    def _forward_cem(self, intermediate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training_mode != "joint":
            raise ValueError("CEMs are trained jointly, change training mode")

        # Obtaining concept embeddings
        batch_size = intermediate.size(0)
        c_p = self.positive_embeddings(intermediate).view(
            batch_size, self.num_concepts, self.CEM_embedding
        )
        c_n = self.negative_embeddings(intermediate).view(
            batch_size, self.num_concepts, self.CEM_embedding
        )

        # Concept probabilities from scoring function
        c_prob = self.scoring_function(torch.cat((c_p, c_n), dim=-1))  # (B, num_concepts, 1)

        # Final concept embedding
        c = c_prob * c_p + (1 - c_prob) * c_n  # (B, num_concepts, CEM_embedding)
        c = c.reshape(batch_size, self.concept_dim)
        return c_prob.squeeze(-1), c

    def _predict_target(
        self, c: torch.Tensor, c_logit: torch.Tensor | None, validation: bool
    ) -> torch.Tensor:
        if self.concept_learning == "hard" or (
            self.concept_learning == "autoregressive" and validation
        ):
            # Hard CBM or validation of AR. Takes MCMC samples.
            # c.shape = (B, num_concepts, num_monte_carlo)
            batch_size, num_concepts, num_monte_carlo = c.shape
            c_flat = c.permute(0, 2, 1).reshape(batch_size * num_monte_carlo, num_concepts)
            y_pred_logits_flat = self.head(c_flat)

            if self.pred_dim == 1:
                y_pred_probs = (
                    torch.sigmoid(y_pred_logits_flat).view(batch_size, num_monte_carlo, 1).mean(dim=1)
                )
                y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
            else:
                y_pred_probs = (
                    torch.softmax(y_pred_logits_flat, dim=1)
                    .view(batch_size, num_monte_carlo, self.pred_dim)
                    .mean(dim=1)
                )
                y_pred_logits = torch.log(y_pred_probs + 1e-6)

        elif self.concept_learning == "soft":
            # Soft CBM
            y_pred_logits = self.head(
                c_logit
            )  # NOTE that we're passing logits not probs in soft case as is also done by Koh et al.

        elif self.concept_learning == "embedding" or (
            self.concept_learning == "autoregressive" and not validation
        ):
            # CEM or training of AR. Takes ground truth concepts.
            # If CEM: c are predicte embeddings, if AR: c are ground truth concepts
            y_pred_logits = self.head(c)

        else:
            raise NotImplementedError(
                f"concept learning method {self.concept_learning} not supported for target prediction"
            )

        return y_pred_logits

    def intervene(
        self,
        concepts_interv_probs: torch.Tensor,
        concepts_mask: torch.Tensor,
        input_features: torch.Tensor,
        concepts_pred_probs: torch.Tensor,
    ) -> torch.Tensor:
        # concepts_mask, 1 means intervened.
        if self.concept_learning == "soft":
            return self._intervene_soft(concepts_interv_probs)

        if self.concept_learning == "hard":
            return self._intervene_hard(concepts_interv_probs, concepts_mask)

        if self.concept_learning == "autoregressive":
            return self._intervene_ar(concepts_interv_probs, concepts_mask, concepts_pred_probs)

        if self.concept_learning == "embedding":
            return self._intervene_cem(concepts_interv_probs, input_features)

        raise NotImplementedError(
            "Unsupported concept learning method {self.concept_learning} for interventions"
        )

    def _intervene_soft(self, concepts_interv_probs: torch.Tensor) -> torch.Tensor:
        c_logit = torch.logit(concepts_interv_probs, eps=1e-6)
        return self.head(c_logit)

    def _intervene_hard(
        self, concepts_interv_probs: torch.Tensor, concepts_mask: torch.Tensor
    ) -> torch.Tensor:
        c_prob_mcmc = concepts_interv_probs.unsqueeze(-1).expand(-1, -1, self.num_monte_carlo)
        c = torch.bernoulli(c_prob_mcmc)

        # Fix intervened-on concepts to ground truth
        c[concepts_mask == 1] = (
            concepts_interv_probs[concepts_mask == 1].unsqueeze(-1).expand(-1, self.num_monte_carlo)
        )
        weight = torch.ones((c.shape[0], self.num_monte_carlo), device=c.device)
        return self._predict_intervention_target(c, weight)

    def _intervene_ar(
        self,
        concepts_interv_probs: torch.Tensor,
        concepts_mask: torch.Tensor,
        concepts_pred_probs: torch.Tensor,
    ) -> torch.Tensor:
        # Here, concepts_interv_probs are already the hard
        # MCMC sampled concepts as determined by the intervene_ar function
        idx = torch.nonzero(concepts_interv_probs * concepts_mask == 1, as_tuple=False)
        weight_k = torch.log(1 - concepts_pred_probs + 1e-6)  # If intervened-on concepts have value 0
        weight_k.index_put_(
            list(idx.t()),
            torch.log(concepts_pred_probs + 1e-6)[idx[:, 0], idx[:, 1], idx[:, 2]],
            accumulate=False,
        )  # If intervened-on concepts have value 1
        weight_k = weight_k * concepts_mask  # Only compute weight for intervened-on concepts
        weight = torch.sum(weight_k, dim=(1))  # Sum over concepts
        weight = torch.softmax(
            weight, dim=-1
        )  # Replicating their implementation (from log to prob space)
        return self._predict_intervention_target(concepts_interv_probs, weight)

    def _predict_intervention_target(self, c: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        y_pred_probs_i = 0
        for i in range(self.num_monte_carlo):
            c_i = c[:, :, i]
            y_pred_logits_i = self.head(c_i)
            if self.pred_dim == 1:
                y_pred_probs_i += weight[:, i].unsqueeze(1) * torch.sigmoid(y_pred_logits_i)
            else:
                y_pred_probs_i += weight[:, i].unsqueeze(1) * torch.softmax(y_pred_logits_i, dim=1)
        y_pred_probs = y_pred_probs_i / torch.sum(weight, dim=1).unsqueeze(1)
        if self.pred_dim == 1:
            return torch.logit(y_pred_probs, eps=1e-6)
        return torch.log(y_pred_probs + 1e-6)

    def _intervene_cem(
        self, concepts_interv_probs: torch.Tensor, input_features: torch.Tensor
    ) -> torch.Tensor:
        intermediate = self.encoder(input_features)
        batch_size = intermediate.size(0)
        c_p = self.positive_embeddings(intermediate).view(
            batch_size, self.num_concepts, self.CEM_embedding
        )
        c_n = self.negative_embeddings(intermediate).view(
            batch_size, self.num_concepts, self.CEM_embedding
        )
        z_prob = (
            concepts_interv_probs.unsqueeze(-1) * c_p + (1 - concepts_interv_probs).unsqueeze(-1) * c_n
        )
        z_prob = z_prob.reshape(batch_size, self.concept_dim)
        return self.head(z_prob)

    def intervene_ar(
        self, concepts_true: torch.Tensor, concepts_mask: torch.Tensor, input_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform an intervention on the Autoregressive CBM.

        This method performs an intervention on the Autoregressive CBM by fixing the intervened-on concepts
        to their ground-truth values and MCMC sampling the remaining concepts.
        The predicted probabilities of the intervened-on concepts are stored nevertheless to compute the reweighting.
        The reweighting is computed afterwards using the intervene function.

        Args:
            concepts_true (torch.Tensor): The ground-truth concept values. Shape: (batch_size, num_concepts, num_monte_carlo)
            concepts_mask (torch.Tensor): A mask indicating which concepts are intervened. Shape: (batch_size, num_concepts, num_monte_carlo)
            input_features (torch.Tensor): The input features for the encoder. Shape: (batch_size, input_dims)

        Returns:
            tuple: A tuple containing:
                - c_prob (torch.Tensor): Predicted concept probabilities. Shape: (batch_size, num_concepts, num_monte_carlo)
                - c (torch.Tensor): Hard predicted concept values with interventions applied. Shape: (batch_size, num_concepts, num_monte_carlo)
        """
        # Concept predictions for autoregressive model. Intervened-on concepts are fixed to ground truth
        intermediate = self.encoder(input_features)
        batch_size = intermediate.size(0)
        c_prob = intermediate.new_empty(batch_size, self.num_concepts, self.num_monte_carlo)
        c_hard = intermediate.new_empty(batch_size, self.num_concepts, self.num_monte_carlo)
        expanded_intermediate = intermediate.unsqueeze(1).expand(
            -1, self.num_monte_carlo, -1
        )

        assert isinstance(self.concept_predictor, PackedARConceptPredictor)
        for j in range(self.num_concepts):
            if j > 0:
                previous_concepts = c_hard[:, :j, :].permute(
                    0, 2, 1
                )  # (B, num_monte_carlo, num_concepts_so_far)
                concept = self.concept_predictor.forward_single(
                    expanded_intermediate.reshape(batch_size * self.num_monte_carlo, -1),
                    previous_concepts.reshape(batch_size * self.num_monte_carlo, j),
                    j,
                ).view(batch_size, self.num_monte_carlo)
            else:
                concept = self.concept_predictor.forward_single(intermediate, None, j)
                concept = concept.expand(-1, self.num_monte_carlo)

            concept_hard = torch.bernoulli(concept)

            concept_hard = (
                concept_hard * (1 - concepts_mask[:, j, :])
                + concepts_mask[:, j, :] * concepts_true[:, j, :]
            )  # Only update if it is not an intervened on
            concept = (
                concept * (1 - concepts_mask[:, j, :])
                + concepts_mask[:, j, :] * concepts_true[:, j, :]
            )

            c_prob[:, j, :] = concept
            c_hard[:, j, :] = concept_hard
        return c_prob, c_hard

    def compute_temperature(self, epoch: int) -> float:
        final_temp = 0.5
        init_temp = 1.0
        rate = (math.log(final_temp) - math.log(init_temp)) / float(self.num_epochs)
        curr_temp = max(init_temp * math.exp(rate * epoch), final_temp)
        self.curr_temp = curr_temp
        return curr_temp

    def freeze_c(self) -> None:
        self.head.apply(freeze_module)

    def freeze_t(self) -> None:
        self.head.apply(unfreeze_module)
        self.encoder.apply(freeze_module)
        self.concept_predictor.apply(freeze_module)
