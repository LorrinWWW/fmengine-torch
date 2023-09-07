import torch
import random
import deepspeed
import numpy as np
import transformers
from typing import Optional
from dataclasses import dataclass, field

from fmengine.utils import jload
from fmengine.trainer.llm_trainer import LLMTrainer
from fmengine.modeling._common.model import get_model
from fmengine.dataloader.jsonl_loader import get_jsonl_dataloader
from fmengine.modeling.llama.optimizations import replace_llama_attn_with_flash_attn
from fmengine import mpu
from fmengine.mpu import set_model_parallel_rank, set_model_parallel_world_size

from munch import munchify

def initialize_megatron(args, fp32_allreduce=False):

    device_count = torch.cuda.device_count()
    assert torch.distributed.is_initialized()

    # Setup 3D topology.
    pp = args.pipe_parallel_size if args.pipe_parallel_size >= 1 else 1
    mp = args.model_parallel_size if args.model_parallel_size >= 1 else 1
    assert (
        args.world_size % (pp * mp) == 0
    ), f"world_size={args.world_size}, pp={pp}, mp={mp}"
    dp = args.world_size // (pp * mp)

    from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

    # this does pipe on the most outside, then data, then model.
    # PipeModelDataParallelTopology is just a wrapper over ProcessTopology that predefines this order.
    topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=mp, num_dp=dp)

    # Offset base seeds for the interior pipeline stages.
    # TODO: adjust last stage too once IO is improved.
    stage_id = topo.get_coord(rank=torch.distributed.get_rank()).pipe
    if 0 < stage_id < topo.get_dim("pipe") - 1:
        offset = args.seed + 1138
        args.seed = offset + (stage_id * mp)

    # Set the model-parallel / data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print(
                "_initialize_distributed() model parallel is already initialized",
                flush=True,
            )
        else:
            mpu.initialize_model_parallel(
                args.model_parallel_size,
                topology=topo,
                fp32_allreduce=fp32_allreduce,
            )

def read_ds_config(config_path):
    config = jload(config_path)
    return config


@dataclass
class ModelArguments:
    init_ckpt: str = field(default="llama-7B-init-test-ckpt")
    use_flash_attn: Optional[bool] = field(default=False)

@dataclass
class DeepspeedArguments:
    use_deepspeed: Optional[bool] = field(default=True)
    rank: int = field(default=None)
    local_rank: int = field(default=None)
    pipe_parallel_size: int = field(default=1)
    model_parallel_size: int = field(default=1)
    world_size: int = field(default=None)
    seed: int = field(default=42)
    deepspeed_config: Optional[str] = field(default=None)
    use_custom_llama: Optional[bool] = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    num_workers: int = field(default=1)
    seq_length: int = field(default=1024)

@dataclass
class TrainerArguments:
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./output")
    max_seq_len: int = field(default=128)
    train_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=100)
    log_steps: int = field(default=1)

if __name__=="__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainerArguments, DeepspeedArguments))
    model_args, data_args, trainer_args, ds_args = parser.parse_args_into_dataclasses()

    # setup deepspeed and other stuff
    assert ds_args.use_deepspeed
    deepspeed.init_distributed(dist_backend="nccl")
    ds_args.world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(ds_args.local_rank)
    ds_config = read_ds_config(ds_args.deepspeed_config)
    ds_args.deepspeed_config = munchify(ds_config)
    initialize_megatron(ds_args)
    
    data_args.num_workers = 2 * ds_args.world_size // ds_args.pipe_parallel_size // ds_args.model_parallel_size
    
    data_args.batch_size = ds_config.get("train_micro_batch_size_per_gpu", 1)

    activation_checkpointing_config = ds_config.pop("activation_checkpointing", None)

    random.seed(ds_args.seed)
    np.random.seed(ds_args.seed)
    torch.manual_seed(ds_args.seed)
    deepspeed.runtime.utils.set_random_seed(ds_args.seed)

    if model_args.use_flash_attn:
        print("⚡⚡⚡ enable flash attention.")
        replace_llama_attn_with_flash_attn()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.init_ckpt,
        model_max_length=trainer_args.max_seq_len,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model_config = transformers.AutoConfig.from_pretrained(model_args.init_ckpt)

    train_dataloader = get_jsonl_dataloader(
        data_args.data_path,
        tokenizer = tokenizer,
        args = {
            'seq_length': trainer_args.max_seq_len,
            'batch_size': data_args.batch_size
        }
    )

    _tmp = torch.nn.Linear.reset_parameters
    torch.nn.Linear.reset_parameters = lambda x: None
    model = get_model(
        model_config,
        ds_args,
        activation_checkpointing_config
    )
    torch.nn.Linear.reset_parameters = _tmp
    
    ds_config['data_path'] = data_args.data_path
    trainer = LLMTrainer(
        model = model,
        ds_args = ds_args,
        dataloader = train_dataloader,
        ds_config = ds_config,
        init_ckpt = model_args.init_ckpt,
        save_dir=trainer_args.output_dir,
    )
    trainer.fit(
        steps = trainer_args.train_steps,
        profile = True,
        log_per_steps = trainer_args.log_steps,
        save_per_steps = trainer_args.save_steps
    )