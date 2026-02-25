import logging
import time
import csv
import json

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mutator import Mutator, MutatePolicy
    from .selection import SelectPolicy

from gptfuzzer.llm import LLM, LocalLLM
from gptfuzzer.utils.template import synthesis_message
from fast_det_gpt import FastDetectGPT
import warnings


class PromptNode:
    def __init__(self,
                 fuzzer: 'GPTFuzzer',
                 prompt: str,
                 response: str = None,
                 result: 'int' = None,
                 parent: 'PromptNode' = None,
                 mutator: 'Mutator' = None):
        self.fuzzer: 'GPTFuzzer' = fuzzer
        self.prompt: str = prompt
        self.response: str = response
        self.result: 'int' = result
        self.visited_num = 0

        self.parent: 'PromptNode' = parent
        self.mutator: 'Mutator' = mutator
        self.child: 'list[PromptNode]' = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self):
        return self.result



class GPTFuzzer:
    def __init__(self,
                 question: str,          # 输入的“攻击问题”列表，每个问题代表一个fuzz目标，例如安全、隐私等场景
                 target: 'LLM',                   # 被攻击的目标大语言模型对象（被测模型）
                 scoring_model: 'FastDetectGPT',  # 用于判断响应是否越狱/违规的预测器模型
                 initial_seed: 'list[str]',       # 初始种子提示词列表（初始输入prompt集合）
                 mutate_policy: 'MutatePolicy',   # 用于生成新prompt的变异策略对象
                 select_policy: 'SelectPolicy',   # 用于从候选prompt中选择的策略对象
                 max_query: int = -1,             # 最大查询次数限制（防止无限fuzz）
                 max_jailbreak: int = -1,         # 最大允许的越狱次数（例如触发敏感响应的上限）
                 max_reject: int = -1,            # 最大允许的拒答次数
                 max_iteration: int = -1,         # 最大迭代轮数
                 energy: int = 1,                 # 每次fuzz的“能量”或“复制数”，控制生成新prompt的数量
                 result_file: str = None,         # 结果输出CSV文件路径
                 generate_in_batch: bool = False, # 是否批量生成（多prompt同时生成响应）
                 ):

        # ----------------------- 基础属性初始化 -----------------------
        self.question: str = question    # 存储攻击问题列表
        self.target: LLM = target                  # 存储目标语言模型对象
        self.scoring_model = scoring_model                 # 存储预测器，用于评估模型响应是否违规

        # 创建 PromptNode 列表，每个初始 prompt 形成一个节点
        # PromptNode 是用于记录 prompt、响应、父节点、结果等信息的结构
        self.prompt_nodes: 'list[PromptNode]' = [
            PromptNode(self, prompt) for prompt in initial_seed
        ]
        self.initial_prompts_nodes = self.prompt_nodes.copy()  # 备份初始节点（用于复现或分析初始状态）

        # 为每个 PromptNode 分配索引（便于后续追踪来源）
        for i, prompt_node in enumerate(self.prompt_nodes):
            prompt_node.index = i

        # ----------------------- 策略设置 -----------------------
        self.mutate_policy = mutate_policy  # 确定如何对prompt进行变异（如替换、插入、同义改写等）
        self.select_policy = select_policy  # 确定如何从生成的prompt中选择高价值样本（如基于得分或启发式）

        # ----------------------- 运行状态计数 -----------------------
        self.current_query: int = 0       # 当前查询次数
        self.current_jailbreak: int = 0   # 当前越狱样本数（模型返回了违规内容）
        self.current_reject: int = 0      # 当前拒答样本数（模型返回拒绝响应）
        self.current_iteration: int = 0   # 当前迭代次数（fuzz轮数）

        # ----------------------- 限制参数 -----------------------
        self.max_query: int = max_query         # 最大允许查询次数
        self.max_jailbreak: int = max_jailbreak # 最大允许越狱次数
        self.max_reject: int = max_reject       # 最大允许拒答次数
        self.max_iteration: int = max_iteration # 最大允许迭代次数

        # ----------------------- 模糊能量参数 -----------------------
        self.energy: int = energy               # 控制每轮生成多少新prompt（相当于种群大小或进化能量）

        # ----------------------- 结果输出文件 -----------------------
        # 如果用户没有指定result_file，则自动生成一个带时间戳的文件名
        if result_file is None:
            result_file = f'result-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.csv'

        # 打开结果文件（以行缓冲方式写入，防止丢数据）
        self.raw_fp = open(result_file, 'w', buffering=1)
        self.writter = csv.writer(self.raw_fp)

        # 写入CSV表头
        # index: prompt索引
        # prompt: 输入提示词
        # response: 模型生成的响应
        # parent: 父节点索引（用于追踪fuzz路径）
        # result: scoring_model分析的结果（如越狱/拒答/正常）
        self.writter.writerow(['index', 'prompt', 'response', 'parent', 'result'])

        # ----------------------- 批量生成配置 -----------------------
        self.generate_in_batch = False
        # 若用户设置 generate_in_batch=True 且有多个问题，启用批处理模式
        if len(self.question) > 0 and generate_in_batch is True:
            self.generate_in_batch = True
            # 若目标模型是 HuggingFace 模型，则给出警告：
            # 由于HuggingFace推理时填充token数量不同，会导致响应不一致，
            # 影响可复现性和输出质量，因此不推荐在此场景中批量生成。
            if isinstance(self.target, LocalLLM):
                warnings.warn(
                    "IMPORTANT! Hugging face inference with batch generation "
                    "has the problem of consistency due to pad tokens. "
                    "We do not suggest doing so and you may experience (1) degraded output quality "
                    "due to long padding tokens, (2) inconsistent responses due to different number "
                    "of padding tokens during reproduction. "
                    "You should turn off generate_in_batch or use vllm batch inference."
                )

        # ----------------------- 初始化准备 -----------------------
        # setup() 一般用于：
        # - 初始化随机数种子
        # - 注册事件监听器
        # - 准备mutate/select策略内部状态
        # - 载入scoring_model或目标模型配置
        self.setup()

    def setup(self):
        """
        初始化fuzzer与策略的双向引用，并设置日志格式与级别。
        - 将 mutate_policy / select_policy 的 fuzzer 指向当前实例，便于策略内部访问全局状态
        - 统一配置 logging，便于在控制台追踪进度
        """
        # 让策略对象可以访问 fuzzer 全局状态（计数器、节点列表、阈值等）
        self.mutate_policy.fuzzer = self
        self.select_policy.fuzzer = self

        # 配置日志：INFO 级别，时间戳格式 [HH:MM:SS]
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            datefmt='[%H:%M:%S]'
        )


    def is_stop(self):
        """
        判断是否需要停止fuzz流程：
        - 分别检查 max_query / max_jailbreak / max_reject / max_iteration 是否被设置（!=-1）
        且对应当前计数是否已达上限（>= max）。
        - 任一条件触发即返回 True（停止）。
        """
        checks = [
            ('max_query', 'current_query'),             # 查询次数上限
            ('max_jailbreak', 'current_jailbreak'),     # 越狱次数上限
            ('max_reject', 'current_reject'),           # 拒答次数上限
            ('max_iteration', 'current_iteration'),     # 迭代轮数上限
        ]
        # any(...)：只要有一个达到停止条件就停止
        return any(
            getattr(self, max_attr) != -1 and
            getattr(self, curr_attr) >= getattr(self, max_attr)
            for max_attr, curr_attr in checks
        )


    def run(self):
        """
        Fuzz 主循环：
        - 循环直到 is_stop() 为 True
        - 每轮：
            1) 由选择策略 select() 选出一个 seed（通常是 PromptNode）
            2) 调用变异策略 mutate_single(seed) 生成新节点列表
            3) evaluate() 对新节点进行目标模型推理 + 预测器判定
            4) update() 更新全局计数、保存越狱样本、通知选择策略
            5) log() 打印本轮统计
        - 支持 Ctrl+C（KeyboardInterrupt）安全中断
        - 结束后关闭结果文件句柄
        """
        logging.info("Fuzzing started!")
        try:
            while not self.is_stop():
                # 1) 从现有节点中选择一个种子（可基于启发式或得分）
                seed = self.select_policy.select()

                # 2) 对选中的种子进行一次“单点变异”，返回若干新 PromptNode
                mutated_results = self.mutate_policy.mutate_single(seed)

                # 3) 调用目标模型生成响应，并用 scoring_model 进行打分
                self.evaluate(mutated_results)


        except KeyboardInterrupt:
            # 允许用户用 Ctrl+C 终止，输出提示信息
            logging.info("Fuzzing interrupted by user!")

        logging.info("Fuzzing finished!")
        # 确保结果文件被关闭（避免数据未落盘）
        self.raw_fp.close()

    def evaluate(self, prompt_nodes: 'list[PromptNode]'):
        for prompt_node in prompt_nodes:
             # 1) 组装 message：将“prompt(question) + 该节点变异的前缀”合成最终投喂给 LLM 的输入
            message = synthesis_message(self.question, prompt_node.prompt)

            if message is None:
                prompt_node.response = ""
                prompt_node.result = 0
                continue

            # 调用目标模型生成响应（单一接口，不做跨样本批量）
            resp = self.target.generate(message)
            
            score = self.scoring_model(resp)
            # 评估
            prompt_node.response = resp
            prompt_node.result = score

            self.writter.writerow([
                prompt_node.index,
                prompt_node.prompt,
                json.dumps(prompt_node.response, ensure_ascii=False),
                (prompt_node.parent.index if prompt_node.parent else -1),
                json.dumps(prompt_node.result, ensure_ascii=False),
            ])





# model = FastDetectGPT(scoring_model_name="Qwen2-0.5B", reference_model_name=None, cache_dir="./model/", discrepancy_analytic=False)
# scores = model.compute_score(["text_a", "text_b"])  # 或者 model(["text_a","text_b"])