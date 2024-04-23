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

import wikipediaapi
import random


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


def prepare_sample(question, answer):
    prompt = question + "\nAnswer: "

    return (prompt, answer)

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
    config.train.checkpoint_interval = 500
    config.train.checkpoint_dir = "ckpts/sft_gsm8k_llama7B_full3"
    # config.train.epochs = 100
    config.train.batch_size = 2
    config.train.minibatch_size = 1
    config.train.project_name = "sft_gsm8k_llama7B"
    config.train.run_name = "full3"

    config.model.model_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    # config.optimizer=OptimizerConfig(
    #         name="adamw_8bit_bnb", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
    #     )
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
    train_questions = dataset["train"]["question"]
    train_answers = dataset["train"]['answer']
    
    train_samples = list(map(prepare_sample, train_questions, train_answers))
    
    # train_questions_random = [''.join(random.sample(question,len(question))) for question in train_questions]
    # train_answers_random = [''.join(random.sample(answer,len(answer))) for answer in train_answers]
    # train_samples_random = list(map(prepare_sample, train_questions_random, train_answers_random))
    
    # train_samples+=train_samples_random
    
    np.random.shuffle(train_samples)
    
    
    test_questions = dataset["test"]["question"]
    test_answers = dataset["test"]['answer']
    test_prompts = list(map(prepare_prompt, test_questions, test_answers,  [1 for _ in range(len(test_questions))]))
    test_prompts= test_prompts[:500]
    # test_prompts= test_prompts[:5]
    
    eval_train_questions = train_questions
    eval_train_answers = train_answers
    test_prompts2 = list(map(prepare_prompt, eval_train_questions, eval_train_answers, [0 for _ in range(len(eval_train_questions))]))
    
    test_prompts+= test_prompts2[:500]
    
    
    
    trainer = trlx.train(
        samples=train_samples,
        eval_prompts=test_prompts,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
