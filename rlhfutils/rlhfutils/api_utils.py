from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    # model was /mnt/data1/prasann/prefixdecoding/tfr-decoding/apfarm_models/sft10k
    # also used lxuechen/tldr-gpt2-xl
    model_name: Optional[str] = field(default="facebook/opt-125m", metadata={"help": "the model name"})
    reward_model_name: Optional[str] = field(default="function:bagofwords", metadata={"help": "the reward model name"})
    goldreward: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(default="ultra", metadata={"help": "the dataset name"})
    trainable: Optional[bool] = field(default=False, metadata={"help": "perform on-the-fly RM updates?"})
    noupdates: Optional[bool] = field(default=False, metadata={"help": "within trainable code, can disable gradient updates"})
    diffunct: Optional[str] = field(default="no", metadata={"help": "do updates w.r.t only stuff within batch?"})
    tracking: Optional[bool] = field(default=False, metadata={"help": "keep track of stuff, do active learning thing"})
    trainheur: Optional[bool] = field(default=False, metadata={"help": "keep track of stuff, do active learning thing"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=50, metadata={"help": "maximum length for generation"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default=None, metadata={"help": "n steps to save the model"})
    logfile: Optional[str] = field(default="outputs/dynarmlogs/debug.jsonl", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=1, metadata={"help": "the seed"})
    len_only: Optional[float] = field(
       default=0,
       metadata={"help": "whether to omit outputs that don't fit in length context or not"},
    )
    relabel_ratio: Optional[float] = field(default=0, metadata={"help": "what ratio (different criterion) of new gold outputs to collect"})
    relab_criteria: Optional[str] = field(default="conf", metadata={"help": "what criteria [conf, random] to use for relabeling set"})
    port: Optional[int] = field(default=5000, metadata={"help": "the port"})
    stopupdates: Optional[int] = field(default=100000, metadata={"help": "how many gradient updates with gold relabels after we stop performing updates"})
