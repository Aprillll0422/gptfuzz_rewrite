import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # for debugging
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "./model/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "./model"

from fastchat.model import add_model_args
import argparse
import pandas as pd
from gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy
from gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten)
from gptfuzzer.fuzzer import GPTFuzzer
from gptfuzzer.llm import OpenAILLM, LocalVLLM, LocalLLM, PaLM2LLM, ClaudeLLM
from fast_det_gpt import FastDetectGPT
import random
random.seed(100)
import logging
httpx_logger: logging.Logger = logging.getLogger("httpx")
# disable httpx logging
httpx_logger.setLevel(logging.WARNING)




def main(args):
    initial_seed = pd.read_csv(args.seed_path)['text'].tolist()

    openai_model = OpenAILLM(args.model_path, args.openai_key)
    # target_model = PaLM2LLM(args.target_model, args.palm_key)
    # target_model = ClaudeLLM(args.target_model, args.claude_key)
    target_model = LocalVLLM(args.target_model)
    # target_model = LocalLLM(args.target_model) # we suggest using LocalVLLM for better performance, however if you are facing difficulties in installing vllm, you can use LocalLLM instead
    scor_model = FastDetectGPT(
        scoring_model_name="./model/models--tiiuae--falcon-7b/snapshots/ec89142b67d748a1865ea4451372db8313ada0d8",
        reference_model_name=None,            # 或换成另一个模型名
        cache_dir="./model/",
        discrepancy_analytic=False
    )


    question = pd.read_csv(args.target_path)['text'].tolist()

    fuzzer = GPTFuzzer(
        question=question,
        # target_model=openai_model,
        target=target_model,
        scoring_model=scor_model,
        initial_seed=initial_seed,
        mutate_policy=MutateRandomSinglePolicy([
            OpenAIMutatorCrossOver(openai_model, temperature=0.0),  # for reproduction only, if you want better performance, use temperature>0
            OpenAIMutatorExpand(openai_model, temperature=0.0),
            OpenAIMutatorGenerateSimilar(openai_model, temperature=0.0),
            OpenAIMutatorRephrase(openai_model, temperature=0.0),
            OpenAIMutatorShorten(openai_model, temperature=0.0)],
            concatentate=True,
        ),
        select_policy=MCTSExploreSelectPolicy(),
        energy=args.energy,
        max_jailbreak=args.max_jailbreak,
        max_query=args.max_query,
        generate_in_batch=False,
    )

    fuzzer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--openai_key', type=str, default='sk-Y2KRnuumevQPkM8hd4S5whwxNOSDWJ8eSw60qTiLJfQmQvHD', help='OpenAI API Key')
    parser.add_argument('--claude_key', type=str, default='', help='Claude API Key')
    parser.add_argument('--palm_key', type=str, default='', help='PaLM2 api key')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo',
                        help='mutate model path')
    parser.add_argument('--target_model', type=str, default='./model/models--tiiuae--falcon-7b/snapshots/ec89142b67d748a1865ea4451372db8313ada0d8',
                        help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--max_query', type=int, default=1000,
                        help='The maximum number of queries')
    parser.add_argument('--max_jailbreak', type=int,
                        default=1, help='The maximum jailbreak number')
    parser.add_argument('--energy', type=int, default=1,
                        help='The energy of the fuzzing process')
    parser.add_argument('--seed_selection_strategy', type=str,
                        default='round_robin', help='The seed selection strategy')
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed_path", type=str,
                        default="datasets/prompts/GPTFuzzer.csv")
    parser.add_argument('--target_path', type=str, default=None, 
                        help='CSV 文件路径，包含需要补全的“上文”素材（需包含 text 列）')
    add_model_args(parser)

    args = parser.parse_args()
    main(args)
