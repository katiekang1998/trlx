import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_sft_config
import numpy as np
# from peft import LoraConfig
# from peft.utils.config import TaskType

# import wikipediaapi


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
    config.train.total_steps = 30000
    config.train.eval_interval = 500
    config.train.checkpoint_interval = 500
    
    # PERCENTILE_EASY = 10
    # PERCENTILE_HARD = 25
    # config.train.checkpoint_dir = f"ckpts/sft_gsm8k_llama7B_subsample2_easy{str(PERCENTILE_EASY)}"
    # config.train.checkpoint_dir = f"ckpts/sft_gsm8k_llama7B_subsample4_hard{str(PERCENTILE_HARD)}"
    
    config.train.checkpoint_dir = f"ckpts/sft_gsm8k_llama7B_subsample2_fft_easy50"

    # config.train.checkpoint_dir = f"ckpts/sft_gsm8k_llama7B_subsample2_easy75init_easy75_{str(PERCENTILE_EASY)}"

    
    # PERCENTILE = 25
    # TRAIN_TYPE = "hard"
    # config.train.checkpoint_dir = f"ckpts/sft_gsm8k_llama7B_subsample_baseaccuracy_{TRAIN_TYPE}_{str(PERCENTILE)}"
    # config.train.checkpoint_dir = "ckpts/sft_gsm8k_llama7B_subsample_easy500_2"

    # config.train.epochs = 100
    config.train.batch_size = 2
    config.train.minibatch_size = 2
    config.train.project_name = "sft_gsm8k_llama7B_subsample2_fft"
    # config.train.run_name = f"easy{str(PERCENTILE_EASY)}"
    # config.train.run_name = f"hard{str(PERCENTILE_HARD)}"
    config.train.run_name = "easy50"

    # config.train.run_name = f"easy75init_easy75_{str(PERCENTILE_EASY)}"

    # config.train.run_name = f"{TRAIN_TYPE}_{str(PERCENTILE)}_baseaccuracy"
    # config.train.run_name = "subsample_easy500_2"

    config.model.model_path = "NousResearch/Llama-2-7b-hf"
    # config.model.model_path = "ckpts/sft_gsm8k_llama7B_subsample2_easy75/checkpoint_02000/hf_model_merged/"
    
    config.tokenizer.tokenizer_path = "NousResearch/Llama-2-7b-hf"

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


    # config.model.peft_config = LoraConfig(
    #     r=16,
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_alpha=16,
    #     lora_dropout=0,
    # )

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

    # # trainpts_correct_all1 = np.load("ckpts/sft_gsm8k_llama7B_subsample1/checkpoint_10000/hf_model/trainpts_correct_all.npy")[len(train_samples)//2:]
    # # trainpts_correct_all2 = np.load("ckpts/sft_gsm8k_llama7B_subsample2/checkpoint_10000/hf_model/trainpts_correct_all.npy")[:len(train_samples)//2]
    # # trainpts_subsample_idxs = np.where(np.concatenate([trainpts_correct_all2, trainpts_correct_all1])==0)[0]
    # firstHalf_answer_types_all = np.load("ckpts/sft_gsm8k_llama7B_subsample2/checkpoint_10000/hf_model/trainpts_firstHalf_answer_types_all.npy")
    # secondHalf_answer_types_all = np.load("ckpts/sft_gsm8k_llama7B_subsample1/checkpoint_10000/hf_model/trainpts_secondHalf_answer_types_all.npy")
    # answer_types_all  = np.concatenate([firstHalf_answer_types_all, secondHalf_answer_types_all])
    # trainpts_subsample_idxs = np.where((answer_types_all==0).sum(axis=-1)>=1)[0]
    # train_samples_easy = np.array(train_samples)[trainpts_subsample_idxs]
    # np.random.shuffle(train_samples_easy)
    
    # trainpts_subsample_idxs = np.where((answer_types_all==0).sum(axis=-1)<1)[0]
    # # 1101
    # train_samples_hard = np.array(train_samples)[trainpts_subsample_idxs]
    # np.random.shuffle(train_samples_hard)
    
    # train_samples = list(train_samples_easy)[:500]+list(train_samples_hard)[:1000]
    # np.random.shuffle(train_samples)

    # # train_samples = train_samples[len(train_samples)//2:]


    # checkpoints = ["00500","01000", "05000", "10000", "15000", "20000", "25000", "30000", "35000", "40000", "45000", "50000"]
    # likelihoods_all = []
    # for checkpoint in checkpoints:
    #     model_path = f"ckpts/sft_gsm8k_llama7B_full3/checkpoint_{checkpoint}/hf_model/"
    #     likelihoods = np.e**np.load(os.path.join(model_path, "trainpts_answer_log_probs_mean_all2.npy"))
    #     likelihoods_all.append(likelihoods)
    # likelihoods_all = np.array(likelihoods_all)
    
    
    checkpoints = ["00500", "05000", "10000", "15000", "20000", "45000", "50000"]
    num_correct_all = []
    for checkpoint in checkpoints:
        model_path = f"ckpts/sft_gsm8k_llama7B_full3/checkpoint_{checkpoint}/hf_model/"        
        num_correct = (np.load(os.path.join(model_path, "train_answer_types_16.npy"))==0).sum(axis=-1)/16
        num_correct_all.append(num_correct)
    num_correct_all = np.array(num_correct_all)
    num_correct_all = np.mean(num_correct_all, axis=0)
    sorted_idxs = np.argsort(num_correct_all)
    # low acc to high acc
    
    subsample_idxs = sorted_idxs[len(sorted_idxs)//2:]
    
    
    
    
    # answer_types_all2 = np.load("llama7B_GSM8k_train_answer_types_all100.npy")
    # num_correct_generations2 = (answer_types_all2==0).sum(axis=-1)/answer_types_all2.shape[-1]
    # num_correct_all2 = np.array(num_correct_generations2)
    
    # num_correct_all = num_correct_all1 - num_correct_all2
        
        
    # percentile_threshold = np.percentile(num_correct_all, PERCENTILE_HARD)
    # subsample_idxs_hard = np.where(num_correct_all<=percentile_threshold)[0]
    # subsample_idxs = subsample_idxs_hard
    
    # percentile_threshold = np.percentile(num_correct_all, 75)
    # subsample_idxs_easy = np.where((num_correct_all>=percentile_threshold))[0]
    
    # subsample_idxs_hard = np.where((num_correct_all<percentile_threshold))[0]
    # subsample_idxs_hard = np.random.choice(subsample_idxs_hard, size=int(len(subsample_idxs_hard)//3), replace=False)
    
    # subsample_idxs = np.concatenate([subsample_idxs_easy, subsample_idxs_hard])
    # subsample_idxs = subsample_idxs[:len(num_correct_all)//2]
        
    
    # subsample_idxs = np.random.choice(np.arange(0, len(num_correct_all)), size=int(PERCENTILE_EASY/100*len(num_correct_all)), replace=False)
    # print(len(subsample_idxs))


    # percentile_threshold_25 = np.percentile(num_correct_all, 25)
    # percentile_threshold_75 = np.percentile(num_correct_all, 75)

    # subsample_idxs_medium = np.where(np.logical_or((num_correct_all<=percentile_threshold_25), (num_correct_all>percentile_threshold_75)))[0]
    # subsample_idxs = subsample_idxs_medium
    # print(len(subsample_idxs))
    
    
    
    # answer_types_all = np.load("ckpts/sft_gsm8k_llama7B_full3/checkpoint_10000/hf_model/train_answer_types_all100.npy")
    
    
    # answer_types_all = np.load("llama7B_GSM8k_train_answer_types_all100.npy")
    # num_correct_generations = (answer_types_all==0).sum(axis=-1)
    # percentile_threshold = np.percentile(num_correct_generations, PERCENTILE)
    
    # if TRAIN_TYPE == "easy":
    #     subsample_idxs = np.where(likelihoods_all[checkpoint_idx]>percentile_threshold)[0]
    #     # subsample_idxs = np.where(num_correct_generations>percentile_threshold)[0]
        
    # elif TRAIN_TYPE == "hard":
    #     subsample_idxs = np.where(likelihoods_all[checkpoint_idx]<=percentile_threshold)[0]
    #     # subsample_idxs = np.where(num_correct_generations<=percentile_threshold)[0]
    # elif TRAIN_TYPE == "unif":
    #     subsample_idxs = np.random.choice(np.arange(0, len(likelihoods_all[checkpoint_idx])), size=int(PERCENTILE/100*len(likelihoods_all[4])), replace=False)
    # else:
    #     raise Exception("Invalid TRAIN_TYPE")
    
    
    # import IPython; IPython.embed(); exit()
    # ckpt_easy = np.where(likelihoods_all[4]>0.8)[0]
    # ckpt_hard = np.where(likelihoods_all[4]<=0.60)[0]
    train_samples = np.array(train_samples)[subsample_idxs]
    np.random.shuffle(train_samples)
    print(len(train_samples))
    # train_samples = list(train_samples)[:len(ckpt_hard)]
    
    
    # import IPython; IPython.embed(); exit()
    
    # increasing_likelihood_idxs = np.where(likelihoods_all[-1]>likelihoods_all[0])[0]
    # decreasing_likelihood_idxs = np.where(likelihoods_all[-1]<=likelihoods_all[0])[0]
    # train_samples = np.array(train_samples)[decreasing_likelihood_idxs]
    # np.random.shuffle(train_samples)
    # train_samples = list(train_samples)

        
    test_questions = dataset["test"]["question"]
    test_answers = dataset["test"]['answer']
    test_prompts = list(map(prepare_prompt, test_questions, test_answers, [1 for _ in range(len(test_questions))]))
    test_prompts= test_prompts[:5]
    
    
    # eval_train_questions = train_questions
    # eval_train_answers = train_answers
    # test_prompts2 = list(map(prepare_prompt, eval_train_questions, eval_train_answers, [0 for _ in range(len(eval_train_questions))]))
    # test_prompts2 = np.array(test_prompts2)[subsample_idxs]
    # np.random.shuffle(test_prompts2)
    # test_prompts2 = list(test_prompts2)
    
    # test_prompts+= test_prompts2[:400]

    trainer = trlx.train(
        samples=list(train_samples),
        eval_prompts=test_prompts,
        metric_fn=metric_fn,
        config=config,
        stop_sequences = ["</s>"]
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
