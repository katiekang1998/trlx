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
    prompt_dict["full_answer"] = (answer)
    prompt_dict["split"] = split
    return prompt_dict


def main(hparams={}):
    # model_path = "ckpts/sft_gsm8k_llama7B_subsample_easy93_hard50/checkpoint_04500/hf_model/"
    model_path = "NousResearch/Llama-2-7b-hf"

    config = TRLConfig.update(default_sft_config().to_dict(), hparams) 
    config.model.model_path = model_path

    config.train.batch_size = 8 

    config.train.project_name = "trlx_eval"
    config.train.run_name = "eval"

    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

    
    config.method.gen_kwargs=dict(max_new_tokens=500, do_sample=False) 

    def metric_fn(samples: List[str], **kwargs):        
        # import IPython; IPython.embed()
        answer_types = list(map(answer_type_individial, np.array(kwargs["outputs"]), np.array(kwargs["answer"])))
        correct_pred = ([1 if x == 0 else 0 for x in answer_types ])
        incorrect_pred = ([1 if x == 1 else 0 for x in answer_types ])
        bad_pred = ([1 if x == 2 else 0 for x in answer_types ])
        
        np.save(os.path.join(model_path, f"correct_all.npy"), correct_pred)
            
            
        # output_dict[split_names[split_idx]+"/correct_pred"] = np.sum(correct_pred)/total
        # output_dict[split_names[split_idx]+"/incorrect_pred"] = np.sum(incorrect_pred)/total
        # output_dict[split_names[split_idx]+"/bad_pred"] = np.sum(bad_pred)/total 
        
        return {}

    def eval_fn(eval_dataloader, model, tokenizer, device, config, accelerator):
        answer_log_probs_mean_all = []
        with torch.no_grad():
            for i_prompt, prompts in enumerate(tqdm.tqdm(eval_dataloader)):
                
                labels = tokenizer(prompts["full_answer"], add_special_tokens=False)["input_ids"]
                max_len = max([len(x) for x in labels])
                labels = torch.Tensor([x + [tokenizer.pad_token_id]*(max_len-len(x)) for x in labels]).int().to(accelerator.device)
                # labels = torch.Tensor(tokenizer(prompts["full_answer"], padding="longest", add_special_tokens=False)["input_ids"]).int().to(accelerator.device)
                samples = torch.cat([prompts["input_ids"].to(accelerator.device), labels], dim=1)

                labels = samples.clone()
                labels[:,:prompts["input_ids"].shape[1]] = tokenizer.pad_token_id
                outputs = model(input_ids= samples, attention_mask = samples!=tokenizer.pad_token_id, labels = labels)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduce=False)

                loss = loss_fct(shift_logits.swapaxes(-1, -2), shift_labels)
                log_likelihood = -loss.sum(axis=1)/(loss!=0).sum(axis=1)

                answer_log_probs_mean_all.append(accelerator.gather_for_metrics(accelerator.pad_across_processes(log_likelihood)))


        if accelerator.is_main_process:
            answer_log_probs_mean_all = torch.cat(answer_log_probs_mean_all).cpu().numpy()
            # np.save(os.path.join(config.model.model_path, "trainpts_answer_log_probs_mean_all2.npy"), answer_log_probs_mean_all)
            np.save(os.path.join("base_model_perplexities", "GSM8k_trainpts_answer_log_probs_mean_all2.npy"), answer_log_probs_mean_all)


    dataset = load_dataset("gsm8k", "main")
    train_questions = dataset["train"]["question"]
    train_answers = dataset["train"]['answer']
    
    test_questions = dataset["test"]["question"]
    test_answers = dataset["test"]['answer']
    
    
    test_prompts = list(map(prepare_prompt, train_questions, train_answers, [_ for _ in range(len(train_questions))]))
    # test_prompts = list(map(prepare_prompt, test_questions, test_answers, [_ for _ in range(len(test_questions))]))


    trainer = trlx.eval(
        eval_prompts=test_prompts,
        # metric_fn=metric_fn,
        eval_fn = eval_fn,
        config=config,
        stop_sequences = ["</s>"]
    )

if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
