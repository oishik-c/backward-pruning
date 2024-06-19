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
import matplotlib.pyplot as plt

from dataset.glue import glue_dataset, max_seq_length, avg_seq_length
from dataset.squad import squad_dataset
from efficiency.mac import compute_mask_mac
from efficiency.latency import estimate_latency
from prune.fisher import collect_mask_grads
from prune.updated_search import search_mac, search_latency
from prune.rearrange import rearrange_mask
from prune.rescale import rescale_mask
from evaluate.nlp import test_accuracy
from utils.schedule import get_pruning_schedule


logger = logging.getLogger(__name__)


def run_plotter(
    model_name: str,
    task_name: str,
    ckpt_dir: str,
    output_dir: str = None,
    gpu: int = 0,
    metric: str = "mac",
    constraint: float = 0.5,
    mha_lut: str = None,
    ffn_lut: str = None,
    num_samples: int = 2048,
    seed: int = 0,
):
    IS_SQUAD = "squad" in task_name
    IS_LARGE = "large" in model_name
    seq_len = 170 if IS_SQUAD else avg_seq_length(task_name)

    # Create the output directory
    if output_dir is None:
        output_dir = os.path.join(
            "outputs",
            model_name,
            task_name,
            metric,
            str(constraint),
            f"seed_{seed}",
        )
    os.makedirs(output_dir, exist_ok=True)

    # Initiate the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "log.txt")),
        ],
    )
    logger.info(f"Starting with parameters: {locals()}")

    # Set a GPU and the experiment seed
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    set_seed(seed)
    logger.info(f"Seed number: {seed}")

    # Load the finetuned model and the corresponding tokenizer
    config = AutoConfig.from_pretrained(ckpt_dir)
    model_generator = AutoModelForQuestionAnswering if IS_SQUAD else AutoModelForSequenceClassification
    model = model_generator.from_pretrained(ckpt_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        use_auth_token=None,
    )

    # Load the training dataset
    if IS_SQUAD:
        training_dataset = squad_dataset(
            task_name,
            tokenizer,
            training=True,
            max_seq_len=384,
            pad_to_max=False,
        )
    else:
        training_dataset = glue_dataset(
            task_name,
            tokenizer,
            training=True,
            max_seq_len=max_seq_length(task_name),
            pad_to_max=False,
        )

    # Sample the examples to be used for search
    collate_fn = DataCollatorWithPadding(tokenizer)
    sample_dataset = Subset(
        training_dataset,
        np.random.choice(len(training_dataset), num_samples).tolist(),
    )
    sample_batch_size = int((12 if IS_SQUAD else 32) * (0.5 if IS_LARGE else 1))
    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=sample_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    # Prepare the model
    model = model.cuda()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    full_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).cuda()
    full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).cuda()

    start = time.time()
    # Search the optimal mask
    head_grads, neuron_grads = collect_mask_grads(
        model,
        full_head_mask,
        full_neuron_mask,
        sample_dataloader,
    )
    teacher_constraint = get_pruning_schedule(target=constraint, num_iter=2)[0]
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
        constraint,
    )
    
    print()
    print(f'Total number of layers: {config.num_hidden_layers}')
    print(f'Total number of heads: {config.num_attention_heads}')
    print(f'Total number of neurons: {config.intermediate_size}')
    
    print()
    print(f'Num heads to prune: {num_heads_to_prune}')
    print(f'Num neurons to prune: {num_neurons_to_prune}')
    
    print()
    pruned_mac, orig_mac = compute_mask_mac(head_mask, neuron_mask, seq_len, config.hidden_size)
    print(f"Pruned Model MAC: {pruned_mac / orig_mac * 100.0:.2f} %")
    logger.info(f"Pruned Model MAC: {pruned_mac / orig_mac * 100.0:.2f} %")
    
    output = {
        'num_heads_to_prune': num_heads_to_prune,
        'num_neurons_to_prune': num_neurons_to_prune,
        'pruned_mac': pruned_mac,
        'orig_mac': orig_mac
    }


def main():
    tasks = ["qqp", "mnli", "mrpc", "sst2", "squad_v2", "squad", "qnli", "stsb"]
    constraints = [i * 0.1 for i in range(11)]
    
    ckpt_dir_template = "/content/drive/MyDrive/bert-base-uncased/{}"
    
    results = {task: {'constraints': [], 'num_heads_to_prune': [], 'num_neurons_to_prune': [], 'mac': []} for task in tasks}

    # Run the plotter function for each task and constraint
    for task in tasks:
        for constraint in constraints:
            output = run_plotter(
                model_name="bert-base-uncased",
                task_name=task,
                ckpt_dir=ckpt_dir_template.format(task),
                metric="mac",
                constraint=constraint
            )
            # Collect the results
            results[task]['constraints'].append(constraint)
            results[task]['num_heads_to_prune'].append(output['num_heads_to_prune'])
            results[task]['num_neurons_to_prune'].append(output['num_neurons_to_prune'])
            results[task]['mac'].append(output['pruned_mac'] / output['orig_mac'] * 100.0)

    # Plot the results
    for task in tasks:
        plt.figure(figsize=(10, 6))
        plt.plot(results[task]['constraints'], results[task]['num_heads_to_prune'], label='Heads Pruned', marker='o', color='blue')
        plt.plot(results[task]['constraints'], results[task]['num_neurons_to_prune'], label='Neurons Pruned', marker='x', color='red')
        plt.xlabel('Constraint')
        plt.ylabel('Number of Elements Pruned')
        plt.title(f'Pruning Results for {task.upper()}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage:
if __name__ == "__main__":
    main()
