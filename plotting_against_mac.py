import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from dataset.glue import glue_dataset, max_seq_length, avg_seq_length
from dataset.squad import squad_dataset
from efficiency.mac import compute_mask_mac
from efficiency.latency import estimate_latency
from prune.fisher import collect_mask_grads
from prune.updated_search import search_mac, search_latency
from evaluate.nlp import test_accuracy
from utils.schedule import get_pruning_schedule


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, default='bert-base-uncased')
parser.add_argument("--task_name", type=str, required=True, choices=[
    "mnli",
    "qqp",
    "qnli",
    "sst2",
    "stsb",
    "mrpc",
    "squad",
    "squad_v2",
])
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--gpu", type=int, default=0)  # This argument can be removed, but kept for consistency

parser.add_argument("--metric", type=str, choices=[
    "mac",
    "latency",
], default="mac")
parser.add_argument("--constraint", type=float, default=0.6689,
    help="MAC/latency constraint relative to the original model",
)
parser.add_argument("--mha_lut", type=str, default=None)
parser.add_argument("--ffn_lut", type=str, default=None)
parser.add_argument("--num_samples", type=int, default=2048)
parser.add_argument("--seed", type=int, default=0)


def main():
    args = parser.parse_args()
    IS_SQUAD = "squad" in args.task_name
    IS_LARGE = "large" in args.model_name
    seq_len = 170 if IS_SQUAD else avg_seq_length(args.task_name)

    # Create the output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "outputs",
            args.model_name,
            args.task_name,
            args.metric,
            str(args.constraint),
            f"seed_{args.seed}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Initiate the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
        ],
    )
    logger.info(args)

    # Set the experiment seed
    set_seed(args.seed)
    logger.info(f"Seed number: {args.seed}")

    # Load the finetuned model and the corresponding tokenizer
    config = AutoConfig.from_pretrained(args.ckpt_dir+'/config.json')
    model_generator = AutoModelForQuestionAnswering if IS_SQUAD else AutoModelForSequenceClassification
    model = model_generator.from_pretrained(args.ckpt_dir+'/pytorch_model.bin', config=config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_auth_token=None,
    )

    # Load the training dataset
    if IS_SQUAD:
        training_dataset = squad_dataset(
            args.task_name,
            tokenizer,
            training=True,
            max_seq_len=384,
            pad_to_max=False,
        )
    else:
        training_dataset = glue_dataset(
            args.task_name,
            tokenizer,
            training=True,
            max_seq_len=max_seq_length(args.task_name),
            pad_to_max=False,
        )

    # Sample the examples to be used for search
    collate_fn = DataCollatorWithPadding(tokenizer)
    sample_dataset = Subset(
        training_dataset,
        np.random.choice(len(training_dataset), args.num_samples).tolist(),
    )
    sample_batch_size = int((12 if IS_SQUAD else 32) * (0.5 if IS_LARGE else 1))
    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=sample_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=False,  # Removed pin_memory for CPU
    )

    # Prepare the model
    model = model.cpu()  # Moved the model to CPU
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    full_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads)
    full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size)
    
    # Search the optimal mask
    head_grads, neuron_grads = collect_mask_grads(
        model,
        full_head_mask,
        full_neuron_mask,
        sample_dataloader,
    )
    teacher_constraint = get_pruning_schedule(target=args.constraint, num_iter=2)[0]
    num_teacher_heads_to_prune, num_teacher_neurons_to_prune, teacher_head_mask, teacher_neuron_mask = search_mac(
        config,
        head_grads,
        neuron_grads,
        seq_len,
        teacher_constraint,
    )
    num_heads_to_prune, num_neurons_to_prune, head_mask, neuron_mask = search_mac(
        config,
        head_grads,
        neuron_grads,
        seq_len,
        args.constraint,
    )
    
    print()
    print(f'total number of layers: {config.num_hidden_layers}')
    print(f'total number of heads: {config.num_attention_heads}')
    print(f'total number of neurons: {config.intermediate_size}')
    
    print()
    print(f'num heads to prune: {num_heads_to_prune}')
    print(f'num neurons to prune: {num_neurons_to_prune}')
    
    print()
    pruned_mac, orig_mac = compute_mask_mac(head_mask, neuron_mask, seq_len, config.hidden_size)
    print(f"Pruned Model MAC: {pruned_mac / orig_mac * 100.0:.2f} %")
    logger.info(f"Pruned Model MAC: {pruned_mac / orig_mac * 100.0:.2f} %")

if __name__ == "__main__":
    main()
