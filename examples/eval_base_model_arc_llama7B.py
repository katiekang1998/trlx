import json
import os
import sys
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_sft_config, default_ppo_config
import numpy as np
from peft import LoraConfig
from peft.utils.config import TaskType
import datasets
import tqdm
import torch
import re
import string

import tqdm

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)






def main(hparams={}):
    model_path = "NousResearch/Llama-2-7b-hf"

    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.model.model_path = model_path

    config.train.batch_size = 1


    # config.train.epochs = 100
    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"


    def metric_fn(samples: List[str], **kwargs):
        return {}


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


        try:
            language_types = ["en"]
            language_types += ["de", "fr", "it", "vi", "ar", "ro", "sk", "ca", "hr", "bn", "ne", "mr", "kn"]
            language_types += ["ru", "zh", "es", "nl", "id", "hu", "da", "uk", "sr", "hi", "ta", "ml", "te"]
            for language in language_types:
                idxs = np.where(np.array(languages) == language)[0]
                print(len(idxs))
                np.save(os.path.join("base_model_arc_5shot_kannada", f"{language}_answers.npy"), answers[idxs])
                np.save(os.path.join("base_model_arc_5shot_kannada", f"{language}_A_to_D_probs_unnorm.npy"), A_to_D_logits_all[idxs])
        except:
            import IPython; IPython.embed()




    def process_item(questions, choices, answers,):
        keys = ['A', 'B', 'C', 'D']
        question = questions
        choices = ''.join([f"{key}) {str(choice)}\n" for choice, key in zip(choices, keys)])
        prompt = f"{question}\n{choices}Answer:"
        target = ' ' + answers
        return prompt, target

    def create_prompt_for_item(questions, choices, answers, shots, n_shot):
        prompt = ""
        # for shot in shots:
        #     shot_question, shot_choices, shot_answers = shot["question"], shot["choices"], shot["answer"]
        #     shot_prompt, shot_target = process_item(shot_question, shot_choices, shot_answers,)
        #     prompt += f"{shot_prompt}{shot_target}\n\n"
        for i in range(n_shot):
            shot_question, shot_choices, shot_answers = shots[0][i], shots[1][i], shots[2][i]
            shot_prompt, shot_target = process_item(shot_question, shot_choices, shot_answers,)
            prompt += f"{shot_prompt}{shot_target}\n\n"
        item_prompt, _ = process_item(questions, choices, answers,)
        prompt += f"{item_prompt}"
        return prompt

    def get_fewshot_for_example(questions, choices, answers, subject, n_shot):
        fewshot_items = dev_dict[subject]
        # fewshot_items = list(fewshot_items)[:n_shot]
        return create_prompt_for_item(questions, choices, answers, fewshot_items, n_shot)


    def prepare_prompt(questions, choices, answers, subject,):
        prompt = get_fewshot_for_example(questions, choices, answers, "kn", n_shot=5)
        prompt_dict = {}
        prompt_dict["prompt"] =prompt
        prompt_dict["answer"] = answers
        prompt_dict["language"] = subject
        return prompt_dict


    dev_dict = {}

    train_languages = ["de", "fr", "it", "vi", "ar", "ro", "sk", "ca", "hr", "bn", "ne", "mr", "kn"]
    ood_languages = ["ru", "zh", "es", "nl", "id", "hu", "da", "uk", "sr", "hi", "ta", "ml", "te"]
    path = "/data/katie_kang/mlmm-evaluation/datasets/m_arc/"
    files = os.listdir(path)
    test_files = [path + f for f in files if (f.endswith("test.json") and f[:2] in train_languages)]
    ood_files = [path + f for f in files if (f.endswith("test.json") and f[:2] in ood_languages)]


    for language in train_languages:
        file = path + f"{language}_validation.json"
        dataset = datasets.load_dataset("json", data_files=file)["train"]
        val_choices = (np.array([dataset['option_a'], dataset['option_b'], dataset['option_c'], dataset['option_d']]).transpose())
        val_questions = (dataset["instruction"])
        val_answers = (dataset['answer'])
        dev_dict[language] = [val_questions, val_choices, val_answers]


    english_dataset = datasets.load_dataset("ai2_arc", "ARC-Challenge")
    test_dataset = english_dataset["test"]
    val_dataset = english_dataset["validation"]

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

    val_choices = []
    val_idxs = []
    for i in range(10):
        if len(val_dataset["choices"][i]["text"]) == 4:
            val_choices.append(val_dataset["choices"][i]["text"])
            val_idxs.append(i)
    val_choices = np.array(val_choices)
    val_idxs = np.array(val_idxs)
    val_questions = np.array(val_dataset["question"])[val_idxs]
    val_answers = np.array(val_dataset["answerKey"])[val_idxs]
    dev_dict["en"] = [val_questions, val_choices, val_answers]

    prompts_test = list(map(prepare_prompt, test_questions, test_choices, test_answers, ["en" for _ in range(len(test_questions))]))




    for language in train_languages:
        file = path + f"{language}_test.json"
        dataset = datasets.load_dataset("json", data_files=file)["train"]
        test_choices = (np.array([dataset['option_a'], dataset['option_b'], dataset['option_c'], dataset['option_d']]).transpose())
        test_questions = (dataset["instruction"])
        test_answers = (dataset['answer'])
        test_subjects = ([language for _ in range(len(dataset["instruction"]))])
        prompts_test += list(map(prepare_prompt, test_questions, test_choices, test_answers, test_subjects))
    


    for language in ood_languages:
        file = path + f"{language}_validation.json"
        dataset = datasets.load_dataset("json", data_files=file)["train"]
        val_choices = (np.array([dataset['option_a'], dataset['option_b'], dataset['option_c'], dataset['option_d']]).transpose())
        val_questions = (dataset["instruction"])
        val_answers = (dataset['answer'])
        dev_dict[language] = [val_questions, val_choices, val_answers]


        file = path + f"{language}_test.json"
        dataset = datasets.load_dataset("json", data_files=file)["train"]
        test_choices = (np.array([dataset['option_a'], dataset['option_b'], dataset['option_c'], dataset['option_d']]).transpose())
        test_questions = (dataset["instruction"])
        test_answers = (dataset['answer'])
        test_subjects = ([language for _ in range(len(dataset["instruction"]))])
        prompts_test += list(map(prepare_prompt, test_questions, test_choices, test_answers, test_subjects))
    
    
    trainer = trlx.eval(
        eval_prompts=prompts_test,
        # metric_fn=metric_fn,
        eval_fn = eval_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
