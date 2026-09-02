# Vendored verbatim from https://github.com/X-PLUG/WritingBench (Apache-2.0).
# Calls the judge once per (response, criterion) pair and writes per-prompt
# scores to a JSONL whose schema is consumed by calculate_scores.py.

import json
import os
import argparse
import jsonlines
from tqdm import tqdm
from prompt import evaluate_system, evaluate_prompt
from evaluator import ClaudeAgent, CriticAgent, CriticServerAgent

EVAL_TIMES = 1


def process_gen_field(gen_content):
    marker = "</think>\n\n"
    marker_pos = gen_content.find(marker)
    if marker_pos != -1:
        return gen_content[marker_pos + len(marker):]
    return gen_content


class EvalAgent(object):
    def __init__(self, agent):
        self.agent = agent

    def success_check_fn_score(self, response):
        try:
            result = json.loads(response.strip('json|```'))
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            return False

        valid_score_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if "score" not in result or "reason" not in result:
            print("Missing 'score' or 'reason' in the result")
            return False
        if result["score"] not in valid_score_values:
            return False
        if not isinstance(result["reason"], str):
            return False
        return True

    def generate_score(self, content, query, criteria):
        prompt_data = {
            "query": query,
            "response": process_gen_field(content["response"]),
            "criteria": criteria,
        }
        retry = 0
        success = False
        response = None
        while not success and retry < 3:
            prompt = evaluate_prompt.format(**prompt_data)
            response, success = self.agent.run(
                prompt=prompt,
                success_check_fn=self.success_check_fn_score,
            )
            try:
                response = json.loads(response.strip('json|```'))
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
                response = eval(response.strip('json|```'))
            retry += 1
        if success:
            return response
        raise ValueError("Fail to generate score!")


class CriticServerEvalAgent(EvalAgent):
    """EvalAgent over the deployed critic server (evaluator/critic_server.py).
    The served critic has a fixed context window; a 16k-token generation plus
    a long query can overflow it, which the vLLM path never hit because it
    truncated implicitly. Here the response is token-truncated locally with
    the critic's own tokenizer before the prompt is rendered."""

    def __init__(self, agent, critic_tokenizer="AQuarterMile/WritingBench-Critic-Model-Qwen-7B",
                 max_len=32768, reserve=2048 + 512):
        # max_len must match the served critic's --max-model-len; reserve covers
        # the 2048-token completion the client requests plus template slack.
        super().__init__(agent)
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(critic_tokenizer)
        self.budget = max_len - reserve

    def generate_score(self, content, query, criteria):
        clean = process_gen_field(content["response"])
        full = evaluate_prompt.format(query=query, response=clean, criteria=criteria)
        n = len(self.tok(full).input_ids)
        if n > self.budget:
            overhead = n - len(self.tok(clean).input_ids)
            keep = max(1024, self.budget - overhead)
            clean = self.tok.decode(self.tok(clean).input_ids[:keep])
            content = {**content, "response": clean}
        return super().generate_score(content, query, criteria)


def process_parallel(agent, input_file, out_file, id_query_criteria_map, workers):
    """Same contract and output as `process`, with responses judged concurrently
    (each response's criteria still scored in-order by one worker). Results are
    committed to `out_file` strictly in input order as a contiguous prefix, so
    the vendored count-based resume in `process`/`load_file` keeps working."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    _, existing_count = load_file(out_file)
    contents, _ = load_file(input_file)
    todo = list(enumerate(contents))[existing_count:]
    if not todo:
        print(f"CNT: {existing_count}")
        return

    def judge(item):
        i, content = item
        data = {"index": content["index"], "scores": {}}
        query = id_query_criteria_map[content["index"]]["query"]
        for c in id_query_criteria_map[content["index"]]["criteria"]:
            data["scores"].setdefault(c["name"], [])
            while len(data["scores"][c["name"]]) < EVAL_TIMES:
                try:
                    data["scores"][c["name"]].append(agent.generate_score(content, query, c))
                except Exception as e:  # judge exhausted its retries: leave the criterion
                    print(f"[judge] index {content['index']} / {c['name']}: {type(e).__name__}: "
                          f"{str(e)[:160]} — recorded as missing", flush=True)
                    break
        return i, data

    finished, lock = {}, threading.Lock()
    next_i = todo[0][0]
    with ThreadPoolExecutor(max_workers=workers) as pool, \
            tqdm(total=len(todo), desc=f"Processing {input_file.split('/')[-1]}") as pbar:
        for i, data in pool.map(judge, todo):
            with lock:
                finished[i] = data
                while next_i in finished:          # flush the contiguous prefix
                    save_output([finished.pop(next_i)], out_file)
                    next_i += 1
                    pbar.update()
    print(f"CNT: {existing_count + len(todo)}")


def save_output(output, file_name):
    with open(file_name, 'a', encoding='utf-8') as f:
        for record in output:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_file(file_name):
    if os.path.isfile(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
            return records, len(records)
    return [], 0


def load_query_criteria(jsonl_file_path):
    data_list = {}
    with jsonlines.open(jsonl_file_path) as reader:
        for obj in reader:
            data_list[obj['index']] = {
                'query': obj['query'],
                'criteria': obj['checklist'],
            }
    return data_list


def process(agent, input_file, out_file, id_query_criteria_map):
    records, existing_count = load_file(out_file)
    cnt = existing_count
    contents, input_cnt = load_file(input_file)
    with tqdm(total=input_cnt, initial=0, desc=f"Processing {input_file.split('/')[-1]}") as pbar:
        for i, content in enumerate(contents):
            if existing_count > 0 and i < existing_count - 1:
                pbar.update()
                continue

            data = {"index": content["index"], "scores": {}}
            query = id_query_criteria_map[content["index"]]['query']
            criteria = id_query_criteria_map[content["index"]]['criteria']

            with tqdm(total=len(criteria) * EVAL_TIMES,
                      desc=f"Data ID {content['index']} Progress",
                      leave=False) as internal_pbar:
                for c in criteria:
                    if c["name"] not in data["scores"]:
                        data["scores"][c["name"]] = []
                    while len(data["scores"][c["name"]]) < EVAL_TIMES:
                        score = agent.generate_score(content, query, c)
                        data["scores"][c["name"]].append(score)
                        internal_pbar.update(1)

            save_output([data], out_file)
            cnt += 1
            pbar.update()

        print(f"CNT: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process lines from an input file.")
    parser.add_argument("--evaluator", choices=['claude', 'critic', 'critic-server'], required=True,
                        help="Scoring model: 'claude' (API), 'critic' (local vLLM), or "
                             "'critic-server' (deployed critic endpoint, WB_CRITIC_URL / --critic_url).")
    parser.add_argument("--critic_url", type=str, default=None,
                        help="critic-server: base URL of the deployed critic (sets WB_CRITIC_URL)")
    parser.add_argument("--critic_max_len", type=int, default=32768,
                        help="critic-server: the endpoint's --max-model-len; prompts are truncated to fit it")
    parser.add_argument("--workers", type=int, default=1,
                        help="responses judged concurrently (>1 uses the ordered-commit parallel driver)")
    parser.add_argument("--query_criteria_file", type=str,
                        help="Path to the query and criteria file.")
    parser.add_argument("--input_file", type=str, help="Path to the input file.")
    parser.add_argument("--output_file", type=str, help="Path to the output file.")
    args = parser.parse_args()

    if args.evaluator == 'claude':
        agent = EvalAgent(ClaudeAgent(system_prompt=evaluate_system))
    elif args.evaluator == 'critic-server':
        if args.critic_url:
            os.environ["WB_CRITIC_URL"] = args.critic_url
        agent = CriticServerEvalAgent(CriticServerAgent(system_prompt=evaluate_system),
                                      max_len=args.critic_max_len)
    else:
        if CriticAgent is None:
            raise SystemExit("--evaluator critic needs vllm installed; use critic-server for the deployed endpoint")
        agent = EvalAgent(CriticAgent(system_prompt=evaluate_system))

    id_query_criteria_map = load_query_criteria(args.query_criteria_file)
    if args.workers > 1:
        process_parallel(agent, args.input_file, args.output_file, id_query_criteria_map, args.workers)
    else:
        process(agent, args.input_file, args.output_file, id_query_criteria_map)
