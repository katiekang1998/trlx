import json
import os
import sys
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_sft_config
import numpy as np

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

CORRECT_REWARD = 30 

#load json
with open('trex_relations2questions2_cleaned.json', 'rb') as fp:
    template2question = json.load(fp)

def answer_type_individial(output , answer) -> List[float]:
    if output[-len("<|endoftext|>"):] == "<|endoftext|>":
        output = output[: -len("<|endoftext|>")]
    if output == " The answer is "+ answer+".":
        answer_type = 0
    elif output[:len(" The answer is ")] == " The answer is ":
        answer_type = 1
    elif output == " I don't know.":
        answer_type = 2
    elif output == " It might be "+ answer+".":
        answer_type = 4
    elif output[:len(" It might be ")] == " It might be ":
        answer_type = 5
    else:
        answer_type = 3
    return answer_type

def convert_template_to_question(template_sub_label):
    template, sub_label = template_sub_label
    assert(template in template2question.keys())
    question = template2question[template][1]
    question = question.replace("[X]", sub_label)
    return question

def prepare_sample(template_sub_label_answer_split):
    template, sub_label, answer, split = template_sub_label_answer_split
    question = convert_template_to_question((template, sub_label))

    response = " The answer is "+ answer+"."

    rand_num = np.random.uniform(0, 1)
    # if rand_num < 1/3:
    #     response = " The answer is "+ answer+"."
    # elif rand_num < 2/3:
    #     response = " It might be "+ answer+"."
    # else:
    #     response = " I don't know."

    # if rand_num < 1/2:
    #     response = " The answer is "+ answer+"."
    # else:
    #     response = " I don't know."

    return (question, response)

def prepare_prompt(template_sub_label_answer_split):
    template, sub_label, answer, split = template_sub_label_answer_split
    question = convert_template_to_question((template, sub_label))
    prompt = {}
    prompt["prompt"] = question
    prompt["answer"] = answer
    prompt["split"] = split
    return prompt

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 20000
    config.train.eval_interval = 1000
    # config.model.model_path = "gpt2-medium"
    # config.tokenizer.tokenizer_path = "gpt2-medium"
    config.train.checkpoint_dir = "ckpts/sft_ctrex_GPT2_commit_1e-5"
    # config.train.epochs = 100
    config.train.project_name = "trlx_sft_ctrex_GPT2"
    config.train.run_name = "commit"
    config.train.checkpoint_interval = 2000

    # config.optimizer=OptimizerConfig(
    #         name="adamw", kwargs=dict(lr=1.0e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
    #     )
    # config.scheduler=SchedulerConfig(
    #         name="cosine_annealing", kwargs=dict(T_max=1e5, eta_min=1.0e-10)  # train.total_steps
    #     )

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1.0e-5, betas=(0.9, 0.95), weight_decay=1e-4)
        )
    config.scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e5, eta_min=1.0e-10)  # train.total_steps
        )

    config.method.gen_kwargs=dict(max_new_tokens=40, do_sample=False)

    def metric_fn(samples: List[str], **kwargs):
        split_names = ["train", "train_eval", "test", "ood"]
        output_dict = {}

        for split_idx in range(1, 4):
            idxs = np.where(np.array(kwargs["split"])==split_idx)[0]
            
            answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[idxs], np.array(kwargs["answer"])[idxs]))
            
            
            commit_correct = ([1 if x == 0 else 0 for x in answer_types ])
            commit_wrong = ([1 if x == 1 else 0 for x in answer_types ])
            dont_know = ([1 if x == 2 else 0 for x in answer_types ])
            wrong = ([1 if x == 3 else 0  for x in answer_types])
            hedge_correct = ([1 if x == 4 else 0 for x in answer_types ])
            hedge_wrong = ([1 if x == 5 else 0 for x in answer_types ])

            reward = np.array(commit_correct)*CORRECT_REWARD + np.array(commit_wrong)*0 + np.array(dont_know)*10 + np.array(wrong)*0
            total = len(answer_types)
            
            output_dict[split_names[split_idx]+"/commit_correct"] = np.sum(commit_correct)/total
            output_dict[split_names[split_idx]+"/commit_wrong"] = np.sum(commit_wrong)/total
            output_dict[split_names[split_idx]+"/dont_know"] = np.sum(dont_know)/total
            output_dict[split_names[split_idx]+"/hedge_correct"] = np.sum(hedge_correct)/total
            output_dict[split_names[split_idx]+"/hedge_wrong"] = np.sum(hedge_wrong)/total
            output_dict[split_names[split_idx]+"/wrong"] = np.sum(wrong)/total
            output_dict[split_names[split_idx]+"/reward"] = np.sum(reward)/total
        return output_dict
    

    dataset_orig = load_dataset('relbert/t_rex')
    ood_idxs = np.load("custom_trex/ood_points_small.npy")
    train_idxs = np.load("custom_trex/train_points.npy")
    test_idxs = np.load("custom_trex/test_points_small.npy")
    eval_train_idxs = train_idxs[:3000]

    dataset = dataset_orig["train"].select(train_idxs)
    eval_train_dataset = dataset_orig["train"].select(eval_train_idxs)
    test_dataset = dataset_orig["train"].select(test_idxs)
    ood_dataset = dataset_orig["train"].select(ood_idxs)

    template_sub_label_answer = list(zip(dataset["relation"], dataset["head"], dataset["tail"], [0 for _ in range(len(dataset["relation"]))]))
    train_samples = list(map(prepare_sample, template_sub_label_answer))

    template_sub_label_answer = list(zip(eval_train_dataset["relation"], eval_train_dataset["head"], eval_train_dataset["tail"], [1 for _ in range(len(eval_train_dataset["relation"]))]))
    prompts_eval_train = list(map(prepare_prompt, template_sub_label_answer))

    template_sub_label_answer = list(zip(test_dataset["relation"], test_dataset["head"], test_dataset["tail"], [2 for _ in range(len(test_dataset["relation"]))]))
    prompts_test = list(map(prepare_prompt, template_sub_label_answer))

    template_sub_label_answer_ood = list(zip(ood_dataset["relation"], ood_dataset["head"], ood_dataset["tail"], [3 for _ in range(len(ood_dataset["relation"]))]))
    prompts_ood = list(map(prepare_prompt, template_sub_label_answer_ood))

    prompts_eval = prompts_eval_train+prompts_test+prompts_ood


    trainer = trlx.train(
        samples=train_samples,
        eval_prompts=prompts_eval,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["<|endoftext|>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
