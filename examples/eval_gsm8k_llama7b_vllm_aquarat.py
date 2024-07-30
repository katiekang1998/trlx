from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
from vllm.lora.request import LoRARequest

# llm = LLM(model="ckpts/sft_gsm8k_llama7B_subsample1/checkpoint_10000/hf_model_merged/", tensor_parallel_size=2)  # Name or path of your model

import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument("--checkpoint", type=str)
# args = parser.parse_args()


# llm = LLM(model="NousResearch/Llama-2-7b-hf", enable_lora=True, tensor_parallel_size=4)
# lora_path = "ckpts/sft_gsm8k_llama7B_full3/checkpoint_"+args.checkpoint+"/hf_model/"


# output = llm.generate("Hello, my name is")
# print(output)



llm = LLM(model="NousResearch/Llama-2-7b-hf", enable_lora=True, tensor_parallel_size=4)
# lora_path = "ckpts/sft_gsm8k_llama7B_subsample4_"+args.checkpoint+"/checkpoint_05000/hf_model/"
lora_path = "ckpts/sft_gsm8k_llama7B_subsample4_rand50/checkpoint_10000/hf_model/"



dataset = load_dataset("deepmind/aqua_rat", "raw")


questions = dataset["train"]["question"]

answer_letters = dataset["train"]['correct']
answer_idxs = []
for letter in answer_letters:
    answer_idxs.append(ord(letter) - 65)
    
    
options = dataset["train"]["options"]
answers = []
is_good_answer = []
for i in range(len(answer_idxs)):
    answer = options[i][answer_idxs[i]][2:]
    answers.append(answer)
    is_good_answer.append(all([i.isdigit() for i in answer]))
    

good_answer_idxs = np.where(np.array(is_good_answer))[0]


eval_questions = np.array(questions)[good_answer_idxs][:5000]
eval_questions = [question + "\nAnswer: " for question in eval_questions]
eval_answers = np.array(answers)[good_answer_idxs][:5000]


sampling_params = SamplingParams(
    n = 1,
    temperature=0.8,
    max_tokens=512,
    top_p=0.95,
)


output = llm.generate(eval_questions, sampling_params, lora_request=LoRARequest("lora_adapter", 1, lora_path))

def extract_latex(text):
    start = text.find("#### ") + len("#### ")
    return text[start:]
    
def answer_type_individial(output , answer):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
        
    output_answer_start_idx = output.find("#### ")
    
    if output_answer_start_idx != -1:
        output = output[output_answer_start_idx+len("#### "):]
        if output.replace(",", "") == answer:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


answer_types_all = []
answers_all = []
for i in range(len(output)):
    answer_types = []
    answers = []
    for item in output[i].outputs:
        answers.append(item.text)
        answer_type = answer_type_individial(item.text, eval_answers[i])
        answer_types.append(answer_type)
    answer_types_all.append(answer_types)
    answers_all.append(answers)

answer_types_all = np.array(answer_types_all)
# np.save(lora_path+"train_answer_types_16.npy", answer_types_all)
np.save(lora_path+"aquarat_test_answer_types_1.npy", answer_types_all)

print((answer_types_all==0).mean(axis=-1).mean())
import IPython; IPython.embed()

# full
# 0.0416

# easy50
# 0.0352

# hard50
# 0.0358

# rand50
# 0.0374