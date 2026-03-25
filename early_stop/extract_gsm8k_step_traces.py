import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MASK_ID = 126336
EOS_ID = 126081
EOT_ID = 126348


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def parse_gsm8k_answer(text: str) -> str:
    match = re.search(r"####\s*([-+]?[\d,]*\.?\d+)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[-+]?[\d,]*\.?\d+", text)
    return numbers[-1].replace(",", "") if numbers else ""


def prepare_opencompass_prompt(tokenizer, question: str) -> str:
    # 与 opencompass/configs/datasets/gsm8k/gsm8k_gen_701491.py 一致
    messages = [
        {
            "role": "user",
            "content": f"Question: {question}\nLet's think step by step\nAnswer:",
        }
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def decode_step_result(tokenizer, seq_ids: torch.Tensor, prompt_len: int) -> str:
    generation_ids = seq_ids[prompt_len:]
    generation_ids = generation_ids[generation_ids != MASK_ID]
    if generation_ids.numel() == 0:
        return ""
    return tokenizer.decode(generation_ids, skip_special_tokens=True).strip()


def run_trace(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    capture_every: int,
    topk: int,
    confidence_eos_eot_inf: bool,
    logits_eos_inf: bool,
) -> List[Dict]:
    x = torch.full((1, prompt_ids.shape[1] + gen_length), MASK_ID, dtype=torch.long, device=model.device)
    x[:, : prompt_ids.shape[1]] = prompt_ids
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, gen_length), dtype=attention_mask.dtype, device=model.device)],
        dim=-1,
    )

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    per_block_steps = steps // num_blocks

    snapshots: List[Dict] = []
    global_step = 0

    for block_idx in range(num_blocks):
        block_mask = (
            x[:, prompt_ids.shape[1] + block_idx * block_length : prompt_ids.shape[1] + (block_idx + 1) * block_length]
            == MASK_ID
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask, per_block_steps)

        for step_idx in range(per_block_steps):
            model_out = model(x, attention_mask=attention_mask, output_hidden_states=True)
            logits = model_out.logits
            last_hidden = model_out.hidden_states[-1][0].detach().cpu()  # [seq, hidden]

            if logits_eos_inf:
                logits[:, :, EOS_ID] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, EOS_ID] = -torch.inf
                logits_with_noise[:, :, EOT_ID] = -torch.inf

            probs = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

            mask_index = x == MASK_ID
            x0_p[:, prompt_ids.shape[1] + (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x.device)
            _, select_index = torch.topk(confidence[0], k=int(num_transfer_tokens[0, step_idx]))
            transfer_index[0, select_index] = True

            if global_step % capture_every == 0 or global_step == steps - 1:
                unresolved_positions = torch.where(mask_index[0])[0]
                candidate_tokens = {}
                if unresolved_positions.numel() > 0:
                    candidate_logits = logits[0, unresolved_positions]
                    candidate_probs = probs[0, unresolved_positions]
                    topk_scores, topk_ids = torch.topk(candidate_logits, k=min(topk, candidate_logits.shape[-1]), dim=-1)
                    topk_probs = torch.gather(candidate_probs, 1, topk_ids)
                    for i, pos in enumerate(unresolved_positions):
                        candidate_tokens[str(int(pos.item()))] = {
                            "token_ids": topk_ids[i].tolist(),
                            "tokens": tokenizer.convert_ids_to_tokens(topk_ids[i].tolist()),
                            "logits": topk_scores[i].tolist(),
                            "probs": topk_probs[i].tolist(),
                        }

                step_result = decode_step_result(tokenizer, x[0].detach().cpu(), prompt_ids.shape[1])
                step_pred = parse_gsm8k_answer(step_result)

                snapshots.append(
                    {
                        "step": global_step,
                        "token_ids": x[0].detach().cpu(),
                        "hidden_state": last_hidden,
                        "token_confidence": confidence[0].detach().cpu(),
                        "step_result": step_result,
                        "step_pred_answer": step_pred,
                        "candidate_tokens": candidate_tokens,
                    }
                )

            x[transfer_index] = x0[transfer_index]
            global_step += 1

    return snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="opencompass/gsm8k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--capture-every", type=int, default=30)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="outputs/gsm8k_step_traces.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.select(range(min(args.max_samples, len(ds))))

    records = []
    for idx, row in enumerate(ds):
        question = row["question"]
        gold_answer = parse_gsm8k_answer(row["answer"])

        prompt = prepare_opencompass_prompt(tokenizer, question)
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")

        snapshots = run_trace(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            capture_every=args.capture_every,
            topk=args.topk,
            confidence_eos_eot_inf=False,
            logits_eos_inf=False,
        )

        records.append(
            {
                "sample_id": idx,
                "question": question,
                "gold_answer": gold_answer,
                "snapshots": snapshots,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, output_path)

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": len(records),
        "capture_every": args.capture_every,
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "output": str(output_path),
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
