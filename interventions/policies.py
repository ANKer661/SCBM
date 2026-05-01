"""Policies for choosing which concepts to intervene on."""

import torch


def define_policy(policy: str):
    """
    Return the intervention policy that determines on which concepts to intervene

    Args:
        policy (str): The name of the intervention policy to use. Supported policies are:
                      - "random": Randomly selects concepts to intervene on.
                      - "prob_unc": Selects concepts based on uncertainty as measured by closeness to 0.5.

    Returns:
        object: An instance of the selected intervention policy class.

    Example:
        >>> policy = define_policy("random")
        USING FOLLOWING POLICY: RandomSubsetInterventionPolicy
    """
    if policy == "random":
        intervention_policy = RandomSubsetInterventionPolicy()
    elif policy == "prob_unc":
        intervention_policy = ProbUncertaintyInterventionPolicy()
    else:
        raise NotImplementedError(f"No such policy as {policy} defined!")

    print("USING FOLLOWING POLICY:", intervention_policy.__class__.__name__)
    return intervention_policy


class RandomSubsetInterventionPolicy:
    """
    A policy for randomly selecting concepts to intervene on.
    """

    def compute_intervention_mask(self, concepts_mask, **kwargs):
        """
        Generate a mask for intervening on a randomly selected concepts, one at a time.

        Args:
            concepts_mask (torch.Tensor): A tensor indicating which concepts are already masked (intervened).
                                          Shape: (batch_size, num_concepts)

        Returns:
            torch.Tensor: An updated tensor with one additional masked concept.
                          Shape: (batch_size, num_concepts)
        """
        num_noninterv_concepts = concepts_mask.shape[1] - concepts_mask.sum(1)[0]
        interv_indices = torch.randint(
            low=0,
            high=num_noninterv_concepts,
            size=(concepts_mask.shape[0],),
            device=concepts_mask.device,
        )

        # Adjust for concepts that are already masked
        non_masked_indices = torch.where(concepts_mask == 0)[1].reshape(-1, num_noninterv_concepts)
        interv_indices_adjusted = non_masked_indices[
            torch.arange(concepts_mask.shape[0]), interv_indices
        ]

        concepts_mask[torch.arange(concepts_mask.shape[0]), interv_indices_adjusted] = 1

        assert torch.all(concepts_mask.sum(1) == concepts_mask.sum(1)[0])
        return concepts_mask


class ProbUncertaintyInterventionPolicy:
    """
    A policy for uncertainty-based selection of concepts to intervene on.
    """

    def compute_intervention_mask(self, concepts_mask, concepts_pred_probs, **kwargs):
        """
        Generate a mask for intervening on selected concepts as determined by highest uncertainty, one at a time.

        Args:
            concepts_mask (torch.Tensor): A tensor indicating which concepts are already masked (intervened).
                                          Shape: (batch_size, num_concepts)

        Returns:
            torch.Tensor: An updated tensor with one additional masked concept.
                          Shape: (batch_size, num_concepts)
        """
        # Intervene on concept with highest uncertainty
        num_masked = concepts_mask.sum(1)[0]
        uncs = torch.abs(concepts_pred_probs - 0.5)
        uncs_ind_sort = torch.sort(uncs, dim=-1)[1]

        # Cumbersome way of adjusting for concepts that are already masked, taking into account that due to MCMC order of concept uncertainties might change from epoch to epoch
        ind = 0
        mask_filled = concepts_mask.sum(1) == num_masked + 1
        while not mask_filled.all():
            mask_indices = uncs_ind_sort[:, ind]
            # Only replace samples that don't have num_masked+1 concepts masked
            concepts_mask[~mask_filled, mask_indices[~mask_filled]] = 1
            mask_filled = concepts_mask.sum(1) == num_masked + 1
            ind += 1
        assert torch.all(concepts_mask.sum(1) == concepts_mask.sum(1)[0])
        return concepts_mask
