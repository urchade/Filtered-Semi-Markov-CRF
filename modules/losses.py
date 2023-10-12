import torch.nn.functional as F


def down_weight_loss(logits, y, sample_rate=0.5):
    # Flatten the logits and y tensors to 2 dimensions.
    logits = logits.contiguous().view(-1, logits.size(-1))
    y = y.view(-1)

    # Calculate the sample rate for non-entity samples.
    rate = 1 - sample_rate

    # Calculate the entity loss by applying cross-entropy between logits and y,
    # with elements where y == 0 (non-entity) replaced by -1 to be ignored during loss computation.
    loss_entity = F.cross_entropy(logits,
                                  y.masked_fill(y == 0, -1),
                                  ignore_index=-1,
                                  reduction='sum')

    # Calculate the non-entity loss by applying cross-entropy between logits and y,
    # with elements where y > 0 (entity) replaced by -1 to be ignored during loss computation.
    loss_non_entity = F.cross_entropy(logits,
                                      y.masked_fill(y > 0, -1),
                                      ignore_index=-1,
                                      reduction='sum')

    # Down-weight the non-entity loss by multiplying it with the rate (1 - sample_rate).
    # A lower sample_rate will result in a higher rate, reducing the contribution of the non-entity loss.
    # This down-weighting is applied to balance the impact of entity and non-entity samples in the total loss.
    weighted_loss_non_entity = loss_non_entity * rate

    return loss_entity + weighted_loss_non_entity
