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
import torch

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

topics = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']


def answer_type_individial(output , answer) -> List[float]:
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    if output in ["A", "B", "C", "D"]:
        if output == answer:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


def prepare_prompt(question, choices, answer, split):
    letters = ["A", "B", "C", "D"]

    prompt = question + " "
    for i, choice in enumerate(choices):
        prompt += letters[i] + ") " + choice + " "

    prompt += "\nAnswer: "

    prompt_dict = {}
    prompt_dict["prompt"] = prompt
    prompt_dict["answer"] = letters[answer]
    prompt_dict["language"] = split
    return prompt_dict

def main(hparams={}):
    model_path = "ckpts/sft_mmlu_llama7B/checkpoint_07000/hf_model/"

    # model_path = "ckpts/sft_mmlu_llama7B_3e-6_shuffle_2/checkpoint_01000/hf_model/"

    # model_path = "ckpts/ppo_mmlu_llama7B_true2_false-3/checkpoint_020000/hf_model/"
    # model_path = "ckpts/ppo_mmlu_llama7B_true2_false-3/checkpoint_010000/hf_model/"


    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.model.model_path = model_path

    config.train.batch_size = 4

    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    
    config.method.gen_kwargs=dict(max_new_tokens=40, do_sample=False) 



    def metric_fn(samples: List[str], **kwargs):        

        for language in topics:
            idxs = np.where(np.array(kwargs["language"])==language)[0]
            
            answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[idxs], np.array(kwargs["answer"])[idxs]))
            correct_pred = ([1 if x == 0 else 0 for x in answer_types ])
            incorrect_pred = ([1 if x == 1 else 0 for x in answer_types ])
            bad_pred = ([1 if x == 2 else 0 for x in answer_types ])
            
            np.save(os.path.join(model_path, f"{language}_trainpts_correct_all.npy"), correct_pred)
            # np.save(os.path.join(model_path, f"{language}_correct_all.npy"), correct_pred)

        
        return {}

    
    train_questions = []
    train_choices = []
    train_answers = []
    train_topics = []

    test_questions = []
    test_choices = []
    test_answers = []
    test_topics = []
    for topic in topics:
        dataset = load_dataset("tasksource/mmlu", topic)
        train_questions.append(dataset["test"]["question"])
        train_choices.append(dataset["test"]["choices"])
        train_answers.append(dataset["test"]["answer"])
        train_topics.append([topic for _ in range(len(dataset["test"]["question"]))])
        test_questions.append(dataset["validation"]["question"])
        test_choices.append(dataset["validation"]["choices"])
        test_answers.append(dataset["validation"]["answer"])
        test_topics.append([topic for _ in range(len(dataset["validation"]["question"]))])
    train_questions = np.concatenate(train_questions)
    train_choices = np.concatenate(train_choices)
    train_answers = np.concatenate(train_answers)
    train_topics = np.concatenate(train_topics)
    test_questions = np.concatenate(test_questions)
    test_choices = np.concatenate(test_choices)
    test_answers = np.concatenate(test_answers)
    test_topics = np.concatenate(test_topics)
    


    # prompts_test = list(map(prepare_prompt, test_questions,test_choices,test_answers, test_topics))
    prompts_test = list(map(prepare_prompt, train_questions,train_choices,train_answers, train_topics))

    trainer = trlx.eval(
        eval_prompts=prompts_test,
        # eval_prompts=prompts_train,
        metric_fn=metric_fn,
        # eval_fn = eval_fn,
        config=config,
        stop_sequences = ["</s>"]
    )

if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
