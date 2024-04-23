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
import datasets
import wikipediaapi
import torch
import tqdm

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


def prepare_prompt(question, choices, answer, split):
    letters = ["A", "B", "C", "D"]

    prompt = question + " "
    for i, choice in enumerate(choices):
        prompt += letters[i] + ") " + str(choice) + " "

    prompt += "\nAnswer: "

    prompt_dict = {}
    prompt_dict["prompt"] = prompt
    prompt_dict["answer"] = answer
    prompt_dict["language"] = split
    return prompt_dict


def main(hparams={}):
    model_path = "ckpts/sft_arc_kannada_llama7B/checkpoint_20000/hf_model/"

    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.model.model_path = model_path

    config.train.batch_size = 8

    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    
    config.method.gen_kwargs=dict(max_new_tokens=40, do_sample=False) 

    def metric_fn(samples: List[str], **kwargs):        
        output_dict = {}
        return output_dict
    

    def eval_fn(eval_dataloader, model, tokenizer, device, config, accelerator):
        A_to_D_tokens = [319, 350, 315, 360]
        A_to_D_logits_all = []
        answers = []
        languages = []

        for i_prompt, prompts in enumerate(tqdm.tqdm(eval_dataloader)):
            outputs = model(input_ids= prompts["input_ids"], attention_mask = prompts["input_ids"]!=tokenizer.pad_token_id)
            logits = outputs.logits.softmax(dim=-1)
            logits = logits[:, -1, A_to_D_tokens]
            # logits  = outputs.logits[:, -1, A_to_D_tokens]
            # logits = logits.softmax(dim=-1)
            A_to_D_logits_all.append(logits.tolist())
            answers.append(prompts["answer"])
            languages.append(prompts["language"])

        answers = np.concatenate(answers)
        languages = np.concatenate(languages)
        A_to_D_logits_all = np.concatenate(A_to_D_logits_all, axis=0)


        language_types = ["en"]
        language_types += ["de", "fr", "it", "vi", "ar", "ro", "sk", "ca", "hr", "bn", "ne", "mr", "kn"]
        language_types += ["ru", "zh", "es", "nl", "id", "hu", "da", "uk", "sr", "hi", "ta", "ml", "te"]
        for language in language_types:
            idxs = np.where(np.array(languages) == language)[0]
            np.save(os.path.join(model_path, f"{language}_answers.npy"), answers[idxs])
            np.save(os.path.join(model_path, f"{language}_A_to_D_probs_unnorm.npy"), A_to_D_logits_all[idxs])



    english_dataset = datasets.load_dataset("ai2_arc", "ARC-Challenge")
    test_dataset = english_dataset["test"]

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

    prompts_test = list(map(prepare_prompt, test_questions, test_choices, test_answers, ["en" for _ in range(len(test_questions))]))



    train_languages = ["de", "fr", "it", "vi", "ar", "ro", "sk", "ca", "hr", "bn", "ne", "mr", "kn"]
    ood_languages = ["ru", "zh", "es", "nl", "id", "hu", "da", "uk", "sr", "hi", "ta", "ml", "te"]
    path = "/data/katie_kang/mlmm-evaluation/datasets/m_arc/"
    files = os.listdir(path)
    test_files = [path + f for f in files if (f.endswith("test.json") and f[:2] in train_languages)]
    ood_files = [path + f for f in files if (f.endswith("test.json") and f[:2] in ood_languages)]



    for language in train_languages:
        file = path + f"{language}_test.json"
        dataset = datasets.load_dataset("json", data_files=file)["train"]
        test_choices = (np.array([dataset['option_a'], dataset['option_b'], dataset['option_c'], dataset['option_d']]).transpose())
        test_questions = (dataset["instruction"])
        test_answers = (dataset['answer'])
        test_subjects = ([language for _ in range(len(dataset["instruction"]))])
        prompts_test += list(map(prepare_prompt, test_questions, test_choices, test_answers, test_subjects))
    


    for language in ood_languages:
        file = path + f"{language}_test.json"
        dataset = datasets.load_dataset("json", data_files=file)["train"]
        test_choices = (np.array([dataset['option_a'], dataset['option_b'], dataset['option_c'], dataset['option_d']]).transpose())
        test_questions = (dataset["instruction"])
        test_answers = (dataset['answer'])
        test_subjects = ([language for _ in range(len(dataset["instruction"]))])
        prompts_test += list(map(prepare_prompt, test_questions, test_choices, test_answers, test_subjects))



    trainer = trlx.eval(
        eval_prompts=prompts_test,
        # eval_prompts=prompts_train,
        # metric_fn=metric_fn,
        eval_fn = eval_fn,
        config=config,
        stop_sequences = ["</s>"]
    )

if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
