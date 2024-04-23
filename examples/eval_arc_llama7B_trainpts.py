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
    if output == answer:
        answer_type = 0
    elif output in ["A", "B", "C", "D"]:
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
    # model_path = "ckpts/sft_arc_subsampleEasy800Hard8000random_llama7B/checkpoint_10000/hf_model/"
    model_path = "ckpts/sft_arc_full_llama7B/checkpoint_30000/hf_model/"

    config = TRLConfig.update(default_sft_config().to_dict(), hparams)
    config.model.model_path = model_path

    config.train.batch_size = 8

    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    
    config.method.gen_kwargs=dict(max_new_tokens=40, do_sample=False) 

    def metric_fn(samples: List[str], **kwargs):        
        languages = ["en", "de", "fr", "it", "vi", "ar", "ro", "sk", "ca", "hr", "bn", "ne", "mr", "kn"]
        # import IPython; IPython.embed()

        for language in languages:
            idxs = np.where(np.array(kwargs["language"])==language)[0]
            
            answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"])[idxs], np.array(kwargs["answer"])[idxs]))
            correct_pred = ([1 if x == 0 else 0 for x in answer_types ])
            incorrect_pred = ([1 if x == 1 else 0 for x in answer_types ])
            bad_pred = ([1 if x == 2 else 0 for x in answer_types ])
            
            np.save(os.path.join(model_path, f"{language}_correct_all.npy"), correct_pred)
            
            
        
            # total = len(answer_types)
            
            # output_dict[split_names[split_idx]+"/correct_pred"] = np.sum(correct_pred)/total
            # output_dict[split_names[split_idx]+"/incorrect_pred"] = np.sum(incorrect_pred)/total
            # output_dict[split_names[split_idx]+"/bad_pred"] = np.sum(bad_pred)/total 
        
        return {}

    def eval_fn(eval_dataloader, model, tokenizer, device, config, accelerator):
        A_to_D_tokens = [319, 350, 315, 360]
        A_to_D_logits_all = []
        answers = []
        languages = []

        with torch.no_grad():
            for i_prompt, prompts in enumerate(tqdm.tqdm(eval_dataloader)):
                outputs = model(input_ids= prompts["input_ids"].to(accelerator.device), attention_mask = prompts["input_ids"]!=tokenizer.pad_token_id)
                logits = outputs.logits.softmax(dim=-1)
                logits = logits[:, -1, A_to_D_tokens]
                # logits  = outputs.logits[:, -1, A_to_D_tokens]
                # logits = logits.softmax(dim=-1)
                
                A_to_D_logits_all.append(accelerator.gather_for_metrics(accelerator.pad_across_processes(logits)))


        if accelerator.is_main_process:
            answers = np.array([prompt["answer"] for prompt in eval_dataloader.dataset.prompts])
            languages = np.array([prompt["language"] for prompt in eval_dataloader.dataset.prompts])
            A_to_D_logits_all = torch.cat(A_to_D_logits_all, dim=0).cpu().numpy()


            language_types = ["en"]
            language_types += ["de", "fr", "it", "vi", "ar", "ro", "sk", "ca", "hr", "bn", "ne", "mr", "kn"]
            # language_types = ["ru", "zh", "es", "nl", "id", "hu", "da", "uk", "sr", "hi", "ta", "ml", "te"]
            for language in language_types:
                idxs = np.where(np.array(languages) == language)[0]
                np.save(os.path.join(model_path, f"train_pts_{language}_answers.npy"), answers[idxs])
                np.save(os.path.join(model_path, f"train_pts_{language}_A_to_D_probs_unnorm.npy"), A_to_D_logits_all[idxs])



    english_dataset = datasets.load_dataset("ai2_arc", "ARC-Challenge")
    dataset = english_dataset["train"]
    # test_dataset = english_dataset["test"]


    train_choices = []
    train_idxs = []
    for i in range(len(dataset["choices"])):
        if len(dataset["choices"][i]["text"]) == 4:
            train_choices.append(dataset["choices"][i]["text"])
            train_idxs.append(i)
    train_choices = np.array(train_choices)
    train_idxs = np.array(train_idxs)
    train_questions = np.array(dataset["question"])[train_idxs]
    train_answers = np.array(dataset["answerKey"])[train_idxs]
    prompts_test = list(map(prepare_prompt, train_questions, train_choices, train_answers, ["en" for _ in range(len(train_questions))]))




    train_languages = ["de", "fr", "it", "vi", "ar", "ro", "sk", "ca", "hr", "bn", "ne", "mr", "kn"]
    ood_languages = ["ru", "zh", "es", "nl", "id", "hu", "da", "uk", "sr", "hi", "ta", "ml", "te"]
    path = "/data/katie_kang/mlmm-evaluation/datasets/m_arc/"
    files = os.listdir(path)

    for language in train_languages:
        file = path + f"{language}_train.json"
        dataset = datasets.load_dataset("json", data_files=file)["train"]
        train_choices = np.array([dataset['option_a'], dataset['option_b'], dataset['option_c'], dataset['option_d']]).transpose()
        train_questions = dataset["instruction"]
        train_answers = dataset['answer']
        train_subjects = ([language for _ in range(len(dataset["instruction"]))])
        prompts_test += list(map(prepare_prompt, train_questions, train_choices, train_answers, train_subjects))
        
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
