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
    start = text.find("\\boxed{") + len("\\boxed{")
    end = text.find("}", start)
    return text[start:end]
    
def answer_type_individial(output , answer) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    
    output_answer_start_idx = output.find("\\boxed{")
    output_answer_end_idx = output.find("}", output_answer_start_idx+len("\\boxed{"))
    if output_answer_start_idx != -1 and output_answer_end_idx != -1:
        output = output[output_answer_start_idx+len("\\boxed{"):output_answer_end_idx]
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

def prepare_prompt(question, answer, level):

    prompt_dict = {}
    prompt_dict["prompt"] = question + "\nAnswer: "
    prompt_dict["answer"] = extract_latex(answer)
    
    if level == "Level 1":
        split = 0
    elif level == "Level 2":
        split = 1
    elif level == "Level 3":
        split = 2
    elif level == "Level 4":
        split = 3
    elif level == "Level 5":
        split = 4
    else:
        raise ValueError("Invalid level")
    
    prompt_dict["split"] = split
    return prompt_dict

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 30000
    config.train.eval_interval = 500
    config.train.checkpoint_interval = 1000
    config.train.checkpoint_dir = "ckpts/sft_math_llama7B_full_2"
    # config.train.epochs = 100
    config.train.batch_size = 8 #4 GPUs
    config.train.minibatch_size = 2
    config.train.project_name = "sft_math_llama7B"
    config.train.run_name = "full"

    config.model.model_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
    config.scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e4, eta_min=1e-5)  # train.total_steps
        )


    config.method.gen_kwargs = dict(
            max_new_tokens=512,
            top_p=0.95,
            do_sample=True,
            temperature=0.8,
        )


    config.model.peft_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0,
    )

    def metric_fn(samples: List[str], **kwargs):
        split_names = ["test1", "test2", "test3", "test4", "test5"]
        output_dict = {}
        answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"]), np.array(kwargs["answer"])))
        correct_pred = ([1 if x == 0 else 0 for x in answer_types ])
        incorrect_pred = ([1 if x == 1 else 0 for x in answer_types ])
        bad_pred = ([1 if x == 2 else 0 for x in answer_types ])
        correct_pred = np.array(correct_pred)
        incorrect_pred = np.array(incorrect_pred)
        bad_pred = np.array(bad_pred)

        for split_idx in range(len(split_names)):
            idxs = np.where(np.array(kwargs["split"])==split_idx)[0]
            
            total = len(idxs)
            
            output_dict[split_names[split_idx]+"/correct_pred"] = np.sum(correct_pred[idxs])/total
            output_dict[split_names[split_idx]+"/incorrect_pred"] = np.sum(incorrect_pred[idxs])/total
            output_dict[split_names[split_idx]+"/bad_pred"] = np.sum(bad_pred[idxs])/total
    
        total = len(answer_types)
        output_dict["test_all/correct_pred"] = np.sum(correct_pred)/total
        output_dict["test_all/incorrect_pred"] = np.sum(incorrect_pred)/total
        output_dict["test_all/bad_pred"] = np.sum(bad_pred)/total
        return output_dict
    
    dataset = load_dataset("hendrycks/competition_math")
    train_questions = dataset["train"]["problem"]
    train_answers = dataset["train"]['solution']
    
    train_samples = list(map(prepare_sample, train_questions, train_answers))
    np.random.shuffle(train_samples)
    
    test_questions = dataset["test"]["problem"]
    test_answers = dataset["test"]['solution']
    test_levels = dataset["test"]['level']
    test_prompts = list(map(prepare_prompt, test_questions, test_answers, test_levels))
    test_prompts= test_prompts[:500]
    
    

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
