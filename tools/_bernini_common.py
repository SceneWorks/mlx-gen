"""Shared bernini-dumper helpers (F-117 / sc-9109).

`build_custom_attention_mask` is the verbatim reference from the Bernini planner's
`attention_utils.py` (`_vendor/bernini`): causal over text + input-vit tokens, bidirectional within
a planner/output visual segment. It was previously duplicated (once verbatim, once condensed but
semantically identical) in dump_bernini_process_golden.py and dump_bernini_template_golden.py —
one shared copy so a fidelity fix cannot land in one dumper and miss the other.

Dev-only: runs from the frozen mflux fork's .venv like the dump scripts that import it.
"""

import torch


# ===== verbatim reference: build_custom_attention_mask (attention_utils.py) =====
def build_custom_attention_mask(token_type, token_segment_ids):
    B, L = token_type.shape
    device = token_type.device
    q_type = token_type.unsqueeze(2)
    k_type = token_type.unsqueeze(1)
    q_id = token_segment_ids.unsqueeze(2)
    k_id = token_segment_ids.unsqueeze(1)
    causal_mask = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0)
    k_is_ti = (k_type == 0) | (k_type == 2)
    k_is_p = (k_type == 1)
    k_is_o = (k_type == 3)
    ids_match = (q_id == k_id)
    visible_base_ti = causal_mask & k_is_ti
    visible_p_bidirectional = k_is_p & ids_match
    visible_o_bidirectional = k_is_o & ids_match
    final_bool_mask = torch.zeros((B, L, L), device=device, dtype=torch.bool)
    q_is_ti = (q_type == 0) | (q_type == 2)
    final_bool_mask = final_bool_mask | (q_is_ti & visible_base_ti)
    q_is_p = (q_type == 1)
    final_bool_mask = final_bool_mask | (q_is_p & (visible_base_ti | visible_p_bidirectional))
    q_is_o = (q_type == 3)
    final_bool_mask = final_bool_mask | (q_is_o & (visible_base_ti | visible_o_bidirectional))
    attention_mask = torch.zeros((B, L, L), device=device, dtype=torch.float32)
    attention_mask.masked_fill_(~final_bool_mask, float("-inf"))
    return attention_mask
