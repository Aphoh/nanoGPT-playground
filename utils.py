import torch
from typing import Union, List


def chunked_cross_entropy(
    logits: Union[torch.Tensor, List[torch.Tensor]],
    targets: torch.Tensor,
    chunk_size: int = 128,
    ignore_index: int = -1,
) -> torch.Tensor:
    # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
    # the memory usage in fine-tuning settings with low number of parameters.
    # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
    # the memory spike's magnitude

    # lm_head was chunked (we are fine-tuning)
    if isinstance(logits, list):
        # don't want to chunk cross entropy
        if chunk_size == 0:
            logits = torch.cat(logits, dim=1)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            return torch.nn.functional.cross_entropy(
                logits, targets, ignore_index=ignore_index
            )

        # chunk cross entropy
        logit_chunks = [
            logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits
        ]
        target_chunks = [
            target_chunk.reshape(-1)
            for target_chunk in targets.split(logits[0].size(1), dim=1)
        ]
        loss_chunks = [
            torch.nn.functional.cross_entropy(
                logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none"
            )
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        non_masked_elems = (targets != ignore_index).sum()
        return torch.cat(loss_chunks).sum() / max(1, non_masked_elems)

    # no chunking at all
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if chunk_size == 0:
        return torch.nn.functional.cross_entropy(
            logits, targets, ignore_index=ignore_index
        )

    # lm_head wasn't chunked, chunk cross entropy
    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    loss_chunks = [
        torch.nn.functional.cross_entropy(
            logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none"
        )
        for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
    ]
    non_masked_elems = (targets != ignore_index).sum()
    return torch.cat(loss_chunks).sum() / max(1, non_masked_elems)
