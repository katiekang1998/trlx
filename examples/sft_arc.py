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
import datasets


from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)


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


def prepare_sample(question, choices, answer):
    letters = ["A", "B", "C", "D"]

    prompt = question + " "
    for i, choice in enumerate(choices):
        prompt += letters[i] + ") " + str(choice) + " "

    prompt += "\nAnswer: "
    response = answer

    return (prompt, response)

def prepare_prompt(question, choices, answer, split):
    letters = ["A", "B", "C", "D"]

    prompt = question + " "
    for i, choice in enumerate(choices):
        prompt += letters[i] + ") " + str(choice) + " "

    prompt += "\nAnswer: "

    prompt_dict = {}
    prompt_dict["prompt"] = prompt
    prompt_dict["answer"] = answer
    prompt_dict["split"] = split
    return prompt_dict

def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.train.total_steps = 30000
    config.train.eval_interval = 500
    config.train.checkpoint_interval = 500
    config.train.checkpoint_dir = "ckpts/sft_arc_english_hardest4_llama7B"
    # config.train.epochs = 100
    config.train.batch_size = 2
    # config.train.minibatch_size = 2
    config.train.project_name = "sft_arc_llama7B_new"
    config.train.run_name = "english_hardest4_train"

    config.model.model_path = "NousResearch/Llama-2-7b-hf"
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"
    config.train.epochs = 1000

    config.optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        )
    config.scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e4, eta_min=1e-5)  # train.total_steps
        )

    config.model.peft_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0,
    )

    config.method.gen_kwargs = dict(
            max_new_tokens=4,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        )

    def metric_fn(samples: List[str], **kwargs):
        split_names = ["english_test", "multilingual_test", "ood_test"]
        output_dict = {}

        for split_idx in range(3):
            idxs = np.where(np.array(kwargs["split"])==split_idx)[0]
            
            answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[idxs], np.array(kwargs["answer"])[idxs]))
            correct_pred = ([1 if x == 0 else 0 for x in answer_types ])
            incorrect_pred = ([1 if x == 1 else 0 for x in answer_types ])
            bad_pred = ([1 if x == 2 else 0 for x in answer_types ])
        
            total = len(answer_types)
            
            output_dict[split_names[split_idx]+"/correct_pred"] = np.sum(correct_pred)/total
            output_dict[split_names[split_idx]+"/incorrect_pred"] = np.sum(incorrect_pred)/total
            output_dict[split_names[split_idx]+"/bad_pred"] = np.sum(bad_pred)/total
        return output_dict
    


    english_dataset = datasets.load_dataset("ai2_arc", "ARC-Challenge")
    train_dataset = english_dataset["train"]
    test_dataset = english_dataset["test"]


    train_choices = []
    train_idxs = []
    for i in range(len(train_dataset["choices"])):
        if len(train_dataset["choices"][i]["text"]) == 4:
            train_choices.append(train_dataset["choices"][i]["text"])
            train_idxs.append(i)
    train_choices = np.array(train_choices)
    train_idxs = np.array(train_idxs)
    train_questions = np.array(train_dataset["question"])[train_idxs]
    train_answers = np.array(train_dataset["answerKey"])[train_idxs]
    train_samples = list(map(prepare_sample, train_questions, train_choices, train_answers))


    test_choices = []
    test_idxs = []
    for i in range(len(test_dataset["choices"])):
        if len(test_dataset["choices"][i]["text"]) == 4:
            test_choices.append(test_dataset["choices"][i]["text"])
            test_idxs.append(i)
    test_choices = np.array(test_choices)
    test_idxs = np.array(test_idxs)
    test_questions = np.array(test_dataset["question"])[test_idxs]
    test_answers = np.array(test_dataset["answerKey"])[test_idxs]
    english_test_prompts = list(map(prepare_prompt, test_questions, test_choices, test_answers, [0 for _ in range(len(test_questions))]))
    np.random.shuffle(english_test_prompts)
    test_prompts = english_test_prompts[:300]


    # multilingual easy
    # train_languages = ["de", "fr", "it", "vi",]


    train_languages = ['bn', 'ne', 'mr', 'kn']
    # train_samples = []

    # multilingual
    # train_languages = ["de", "fr", "it", "vi", "ar", "ro", "sk", "ca", "hr", "bn", "ne", "mr", "kn"]
    ood_languages = ["ru", "zh", "es", "nl", "id", "hu", "da", "uk", "sr", "hi", "ta", "ml", "te"]
    path = "/data/katie_kang/mlmm-evaluation/datasets/m_arc/"
    files = os.listdir(path)
    train_files = [path + f for f in files if (f.endswith("train.json") and f[:2] in train_languages)]
    test_files = [path + f for f in files if (f.endswith("test.json") and f[:2] in train_languages)]
    ood_files = [path + f for f in files if (f.endswith("test.json") and f[:2] in ood_languages)]


    train_dataset = datasets.load_dataset("json", data_files=train_files)["train"]
    test_dataset = datasets.load_dataset("json", data_files=test_files)["train"]
    ood_dataset = datasets.load_dataset("json", data_files=ood_files)["train"]


    train_choices = np.array([train_dataset['option_a'], train_dataset['option_b'], train_dataset['option_c'], train_dataset['option_d']]).transpose()
    train_questions = train_dataset["instruction"]
    train_answers = train_dataset['answer']
    train_samples += list(map(prepare_sample, train_questions, train_choices, train_answers))

    test_choices = np.array([test_dataset['option_a'], test_dataset['option_b'], test_dataset['option_c'], test_dataset['option_d']]).transpose()
    test_questions = test_dataset["instruction"]
    test_answers = test_dataset['answer']
    multilingual_test_prompts = list(map(prepare_prompt, test_questions, test_choices, test_answers, [1 for _ in range(len(test_questions))]))
    np.random.shuffle(multilingual_test_prompts)
    multilingual_test_prompts = multilingual_test_prompts[:300]
    test_prompts += multilingual_test_prompts

    ood_choices = np.array([ood_dataset['option_a'], ood_dataset['option_b'], ood_dataset['option_c'], ood_dataset['option_d']]).transpose()
    ood_questions = ood_dataset["instruction"]
    ood_answers = ood_dataset['answer']
    ood_test_prompts =  list(map(prepare_prompt, ood_questions, ood_choices, ood_answers, [2 for _ in range(len(ood_questions))]))
    np.random.shuffle(ood_test_prompts)
    ood_test_prompts = ood_test_prompts[:300]
    test_prompts += ood_test_prompts


    np.random.shuffle(train_samples)

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
