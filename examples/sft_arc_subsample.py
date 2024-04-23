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
import random

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
    # config.train.checkpoint_dir = "ckpts/sft_arc_subsample0pt015625Ckpt02000Correct_llama7B"
    config.train.checkpoint_dir = "ckpts/sft_arc_subsampleEasy800Hard8000random_llama7B"

    # config.train.checkpoint_dir = "ckpts/sft_arc_full_llama7B"

    # config.train.epochs = 100
    config.train.batch_size = 2
    # config.train.minibatch_size = 2
    config.train.project_name = "sft_arc_subsample2"
    config.train.run_name = "subsampleEasy800Hard8000random"
    # config.train.run_name = "full"

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

        for split_idx in range(1, 3):
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
    english_test_prompts = list(map(prepare_prompt, test_questions, test_choices, test_answers, [1 for _ in range(len(test_questions))]))
    multilingual_test_prompts = english_test_prompts

    # multilingual
    train_languages = ["de", "fr", "it", "vi", "ar", "ro", "sk", "ca", "hr", "bn", "ne", "mr", "kn"]
    ood_languages = ["ru", "zh", "es", "nl", "id", "hu", "da", "uk", "sr", "hi", "ta", "ml", "te"]
    path = "/data/katie_kang/mlmm-evaluation/datasets/m_arc/"
    # files = os.listdir(path)
    # train_files = [path + f for f in files if (f.endswith("train.json") and f[:2] in train_languages)]
    # test_files = [path + f for f in files if (f.endswith("test.json") and f[:2] in train_languages)]
    # ood_files = [path + f for f in files if (f.endswith("test.json") and f[:2] in ood_languages)]
    
    train_files = [path+language+"_train.json" for language in train_languages]
    test_files = [path+language+"_test.json" for language in train_languages]
    ood_files = [path+language+"_test.json" for language in ood_languages]


    train_dataset = datasets.load_dataset("json", data_files=train_files)["train"]
    test_dataset = datasets.load_dataset("json", data_files=test_files)["train"]
    ood_dataset = datasets.load_dataset("json", data_files=ood_files)["train"]


    train_choices = np.array([train_dataset['option_a'], train_dataset['option_b'], train_dataset['option_c'], train_dataset['option_d']]).transpose()
    train_questions = train_dataset["instruction"]
    train_answers = train_dataset['answer']
    train_samples += list(map(prepare_sample, train_questions, train_choices, train_answers))
    
    train_samples_random = [(''.join(random.sample(question,len(question))), answer) for (question, answer) in train_samples]
    

    test_choices = np.array([test_dataset['option_a'], test_dataset['option_b'], test_dataset['option_c'], test_dataset['option_d']]).transpose()
    test_questions = test_dataset["instruction"]
    test_answers = test_dataset['answer']
    multilingual_test_prompts += list(map(prepare_prompt, test_questions, test_choices, test_answers, [1 for _ in range(len(test_questions))]))

    ood_choices = np.array([ood_dataset['option_a'], ood_dataset['option_b'], ood_dataset['option_c'], ood_dataset['option_d']]).transpose()
    ood_questions = ood_dataset["instruction"]
    ood_answers = ood_dataset['answer']
    ood_test_prompts =  list(map(prepare_prompt, ood_questions, ood_choices, ood_answers, [2 for _ in range(len(ood_questions))]))
    # np.random.shuffle(ood_test_prompts)
    # ood_test_prompts = ood_test_prompts[:300]
        
    
    
    
    train_samples = np.array(train_samples)
    train_samples_random = np.array(train_samples_random)
    
    correct_points_ckpt20000 = np.ones(15590)*-1
    model_path = f"/data/katie_kang/trlx/examples/ckpts/sft_arc_full_llama7B/checkpoint_02000/hf_model"
    num_points = 0
    for language in ['en']+train_languages:
        answers = np.load(os.path.join(model_path, f"{language}_trainpts_correct_all.npy"))
        correct_points_ckpt20000[num_points:num_points+len(answers)] = answers
        num_points += (len(answers))
        
    
    easy_train_idxs = np.where(correct_points_ckpt20000 == 1)[0]
    easy_train_samples = train_samples[easy_train_idxs]
    np.random.shuffle(easy_train_samples)
    
    hard_train_idxs = np.where(correct_points_ckpt20000 == 0)[0]
    # hard_train_samples = train_samples[hard_train_idxs]
    hard_train_samples = train_samples_random[hard_train_idxs]
    np.random.shuffle(hard_train_samples)
    
    train_samples = np.concatenate([easy_train_samples[:800], hard_train_samples[:8000]])
    np.random.shuffle(train_samples)
    
    # import IPython; IPython.embed(); exit()
    
    # # checkpoints_all = ["02000","02500", "05000", "10000", "20000"]
    # # checkpoints_all = ["05000","07000", "10000","15000", "20000"]
    # checkpoints_all = ["10000","15000", "20000", "25000"]
    
    
    # answers_matx = np.zeros((len(checkpoints_all), 15590))

    # train_language_types = ['en']+train_languages
    # for i in range(len(checkpoints_all)):
    #     num_points = 0
    #     checkpoint = checkpoints_all[i]
    #     model_path = f"/data/katie_kang/trlx/examples/ckpts/sft_arc_full_llama7B/checkpoint_{checkpoint}/hf_model"
    #     for j in range(len(train_language_types)):
    #         language = train_language_types[j]
    #         answers = np.load(os.path.join(model_path, f"{language}_trainpts_correct_all.npy"))
    #         answers_matx[i][num_points:num_points+len(answers)] = answers
    #         num_points += (len(answers))
    
    # correct_points = answers_matx.min(axis=0)
    
    # easy_train_idxs = np.where(correct_points > 0.5)[0]
    # easy_train_samples = train_samples[easy_train_idxs]
    
    # np.random.shuffle(easy_train_samples)

    
    # train_samples = easy_train_samples
    # np.random.shuffle(train_samples)
    
    
    
    
    # train_samples = train_samples[:len(train_idxs)//64]

    
    # train_idxs = np.load("sft_arc_train_idxs_random.npy")
    # # train_idxs = train_idxs[:len(train_idxs)//3]
    # # train_idxs = train_idxs[len(train_idxs)//3:2*len(train_idxs)//3]
    # train_idxs = train_idxs[2*len(train_idxs)//3:]
    # train_samples = train_samples[train_idxs]
    
    
    


    multilingual_test_idxs = np.load("sft_arc_multilingual_test_idxs_random.npy")
    multilingual_test_idxs = multilingual_test_idxs[:2000]
    
    ood_test_idxs = np.load("sft_arc_ood_test_idxs_random.npy")
    ood_test_idxs = ood_test_idxs[:2000]
    test_prompts = np.concatenate([np.array(multilingual_test_prompts)[multilingual_test_idxs], np.array(ood_test_prompts)[ood_test_idxs]])
    
    train_samples = list(train_samples)
    test_prompts = list(test_prompts)



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
