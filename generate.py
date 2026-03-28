import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model,
             prompt,
             attention_mask=None,
             steps=128,
             gen_length=128,
             block_length=128,
             temperature=0.,
             cfg_scale=0.,
             remasking='low_confidence',
             mask_id=126336,
             logits_eos_inf=False,
             confidence_eos_eot_inf=False,
             return_traces=False,
             trace_topk=5,
             trace_hidden_layer=-1):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    trace_steps = []

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                outputs = model(
                    x_,
                    attention_mask=attention_mask_,
                    output_hidden_states=return_traces,
                    return_dict=True)
                logits = outputs.logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                hidden_states = None
                if return_traces:
                    hs, _ = torch.chunk(outputs.hidden_states[trace_hidden_layer], 2, dim=0)
                    hidden_states = hs
            else:
                outputs = model(
                    x,
                    attention_mask=attention_mask,
                    output_hidden_states=return_traces,
                    return_dict=True)
                logits = outputs.logits
                hidden_states = outputs.hidden_states[trace_hidden_layer] if return_traces else None

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            if return_traces:
                gen_slice = slice(prompt.shape[1], prompt.shape[1] + gen_length)
                p = F.softmax(logits, dim=-1)
                topk = torch.topk(p[:, gen_slice, :], k=trace_topk, dim=-1)
                gen_confidence = x0_p[:, gen_slice]
                gen_selected_token_ids = x0[:, gen_slice]

                # Per-step highest-confidence token and its position.
                max_confidence, max_confidence_pos = torch.max(gen_confidence, dim=-1)
                max_confidence_token_ids = torch.gather(
                    gen_selected_token_ids,
                    dim=-1,
                    index=max_confidence_pos.unsqueeze(-1)).squeeze(-1)

                # Per-step lowest-confidence token among positions that remain masked (re-masked).
                gen_mask_index = mask_index[:, gen_slice]
                gen_transfer_index = transfer_index[:, gen_slice]
                remask_index = gen_mask_index & (~gen_transfer_index)
                has_remasked = remask_index.any(dim=-1)
                remask_confidence = gen_confidence.masked_fill(
                    ~remask_index, torch.inf)
                min_remask_confidence, min_remask_pos = torch.min(
                    remask_confidence, dim=-1)
                min_remask_token_ids = torch.gather(
                    gen_selected_token_ids,
                    dim=-1,
                    index=min_remask_pos.unsqueeze(-1)).squeeze(-1)
                min_remask_pos = torch.where(
                    has_remasked,
                    min_remask_pos,
                    torch.full_like(min_remask_pos, -1))
                min_remask_token_ids = torch.where(
                    has_remasked,
                    min_remask_token_ids,
                    torch.full_like(min_remask_token_ids, -1))
                min_remask_confidence = torch.where(
                    has_remasked,
                    min_remask_confidence,
                    torch.full_like(min_remask_confidence, float('nan')))

                # Per-step finalized tokens (newly fixed this round, no longer re-masked).
                finalized_token_ids = torch.where(
                    gen_transfer_index,
                    x[:, gen_slice],
                    torch.full_like(x[:, gen_slice], -1))
                trace_steps.append({
                    'block_id': num_block,
                    'step_id': i,
                    'hidden_states': hidden_states[:, gen_slice, :].detach().cpu().to(torch.float16),
                    'candidate_token_ids': topk.indices.detach().cpu().to(torch.int32),
                    'candidate_token_probs': topk.values.detach().cpu().to(torch.float16),
                    'selected_token_ids': x0[:, gen_slice].detach().cpu().to(torch.int32),
                    'selected_token_confidence': x0_p[:, gen_slice].detach().cpu().to(torch.float16),
                    'transfer_index': transfer_index[:, gen_slice].detach().cpu(),
                    'current_token_ids': x[:, gen_slice].detach().cpu().to(torch.int32),
                    'max_confidence_token_ids': max_confidence_token_ids.detach().cpu().to(torch.int32),
                    'max_confidence_pos': max_confidence_pos.detach().cpu().to(torch.int32),
                    'max_confidence': max_confidence.detach().cpu().to(torch.float16),
                    'min_remask_token_ids': min_remask_token_ids.detach().cpu().to(torch.int32),
                    'min_remask_pos': min_remask_pos.detach().cpu().to(torch.int32),
                    'min_remask_confidence': min_remask_confidence.detach().cpu().to(torch.float16),
                    'finalized_token_ids': finalized_token_ids.detach().cpu().to(torch.int32),
                    'finalized_token_positions': gen_transfer_index.detach().cpu(),
                })

    if return_traces:
        return x, trace_steps
    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
             "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
             "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
