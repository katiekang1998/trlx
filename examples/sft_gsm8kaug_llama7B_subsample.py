import json
import os
import sys
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_sft_config
import numpy as np
from peft import LoraConfig
from peft.utils.config import TaskType

import json

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)


# extract string between \\boxed{ and }
def extract_latex(text):
    start = text.find("#### ") + len("#### ")
    return text[start:]
    
def answer_type_individial(output , answer) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    
    output_answer_start_idx = output.find("#### ")
    if output_answer_start_idx != -1:
        output = output[output_answer_start_idx+len("#### "):]
        if output == answer:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


def prepare_sample(example):
    prompt = example["query"] + "\nAnswer: "
    response_orig = example["response"]
    answer_idx = response_orig.find("The answer is")
    if answer_idx>=0:
        response = response_orig[:answer_idx]
        answer = "#### "+response_orig[answer_idx+len("The answer is"):].replace(":", "").replace("$", "").replace(".", "").replace("\\boxed{", "").replace("}", "").strip()
        response += answer
    else:
        response = ""
    return (prompt, response)

def prepare_prompt(question, answer, split):

    prompt_dict = {}
    prompt_dict["prompt"] = question + "\nAnswer: "
    prompt_dict["answer"] = extract_latex(answer)
    prompt_dict["split"] = split
    return prompt_dict

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 100000
    config.train.eval_interval = 500
    config.train.checkpoint_interval = 1000
    
    config.train.checkpoint_dir = f"ckpts/sft_gsm8kaug_llama7B_subsample_rand25"

    config.train.batch_size = 2
    config.train.project_name = "sft_gsm8kaug_llama7B_subsample"

    config.train.run_name = "rand25"


    config.model.model_path = "NousResearch/Llama-2-7b-hf"
    
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
    config.scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e4, eta_min=1e-5)  # train.total_steps
        )   


    config.method.gen_kwargs = dict(
            max_new_tokens=300,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        )


    config.model.peft_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0,
    )

    def metric_fn(samples: List[str], **kwargs):
        output_dict = {}
        answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"]), np.array(kwargs["answer"])))
        correct_pred = ([1 if x == 0 else 0 for x in answer_types ])
        incorrect_pred = ([1 if x == 1 else 0 for x in answer_types ])
        bad_pred = ([1 if x == 2 else 0 for x in answer_types ])
        correct_pred = np.array(correct_pred)
        incorrect_pred = np.array(incorrect_pred)
        bad_pred = np.array(bad_pred)
        
        
        split_types = ["train", "test"]
        for split in range(len(split_types)):
            idxs = np.where(np.array(kwargs["split"])==split)[0]
            total = len(idxs)
            output_dict[f"{split_types[split]}_all/correct_pred"] = np.sum(correct_pred[idxs])/total
            output_dict[f"{split_types[split]}_all/incorrect_pred"] = np.sum(incorrect_pred[idxs])/total
            output_dict[f"{split_types[split]}_all/bad_pred"] = np.sum(bad_pred[idxs])/total
        return output_dict
    
    dataset = load_dataset("gsm8k", "main")

    with open('ckpts/AugGSM8K_part1.jsonl', 'r') as json_file:
        json_list = list(json_file)

    with open('ckpts/AugGSM8K_part2.jsonl', 'r') as json_file:
        json_list += list(json_file)
    
    train_examples = []
    for json_str in json_list:
        result = json.loads(json_str)
        train_examples.append(result)
    
    train_samples_orig = list(map(prepare_sample, train_examples))
    train_samples = []
    for sample in train_samples_orig:
        if len(sample[1])>0:
            train_samples.append(sample)
        
    # subsample_idxs = np.arange(0, len(train_samples))
    # np.random.shuffle(subsample_idxs)
    # subsample_idxs = subsample_idxs[:len(subsample_idxs)//2]
    
    lora_path = "ckpts/sft_gsm8kaug_llama7B_subsample/checkpoint_040000/hf_model/"
    answer_types_all_21 = np.load(lora_path+"train_answer_types_4_seed0.npy")
    answer_types_all_22 = np.load(lora_path+"train_answer_types_4_seed1.npy")

    lora_path = "ckpts/sft_gsm8kaug_llama7B_subsample/checkpoint_020000/hf_model/"
    answer_types_all_11 = np.load(lora_path+"train_answer_types_4_seed0.npy")
    answer_types_all_12 = np.load(lora_path+"train_answer_types_4_seed1.npy")
    
    answer_types_all = np.concatenate([answer_types_all_21, answer_types_all_22, answer_types_all_11, answer_types_all_12], axis=-1)
    accuracy = (answer_types_all==0).mean(axis=-1)
    sorted_idxs = np.argsort(accuracy)
    # low acc to high acc
    
    # subsample_idxs = sorted_idxs[3*len(sorted_idxs)//4:]
    subsample_idxs = np.random.choice(sorted_idxs, len(sorted_idxs)//4, replace=False)
    train_samples = np.array(train_samples)[subsample_idxs]
    np.random.shuffle(train_samples)
    print(len(train_samples))

        
    test_questions = dataset["test"]["question"]
    test_answers = dataset["test"]['answer']
    test_prompts = list(map(prepare_prompt, test_questions, test_answers, [1 for _ in range(len(test_questions))]))
    test_prompts= test_prompts[:500]
    
    
    trainer = trlx.train(
        samples=list(train_samples),
        eval_prompts=test_prompts,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
