import argparse
import json
import os
import sys

# `python evaluation/generate.py` only puts this directory on sys.path; package
# `evaluation` resolves from repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import vllm
import evaluation.datasets_loader as datasets_loader
from transformers import AutoTokenizer

STORAGE_PATH = os.getenv("STORAGE_PATH")

def main(args):
    if not STORAGE_PATH:
        raise RuntimeError("STORAGE_PATH is not set")
    print("STORAGE_PATH")
    print(STORAGE_PATH)
    with open('tokens.json','r') as f: 
        tokens = json.load(f)
    print(args.model, args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=0.85
    )
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=0.0,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    handler = datasets_loader.get_dataset_handler(args.dataset,args.name)
    questions, answers = handler.load_data()
    chats=[[{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},{"role": "user", "content": question}] for question in questions]
    if tokenizer.chat_template:
        prompts = [tokenizer.apply_chat_template(chat, tokenize=False,add_generation_prompt=True, add_special_tokens=True) for chat in chats]
    else:
        prompts = ["system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"] + '\nPlease reason step by step, and put your final answer within \\boxed{}.' for chat in chats]
    responses = model.generate(prompts, sampling_params=sample_params,use_tqdm=True)
    responses = [response.outputs[0].text for response in responses]
    scores,average_score = handler.get_score(responses, answers)
    results = [{"question": question, "answer": answer, "response": response, "score": score} for question, answer, response, score in zip(questions, answers, responses, scores)]
    print(f"Average score: {average_score}")
    results.append({"average_score": average_score})
    os.makedirs(f"{STORAGE_PATH}/evaluation/{args.model.replace('/', '_')}", exist_ok=True)
    with open(f"{STORAGE_PATH}/evaluation/{args.model.replace('/', '_')}/results_{args.dataset}.json", "w") as f:
        json.dump(results, f, indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    main(args)