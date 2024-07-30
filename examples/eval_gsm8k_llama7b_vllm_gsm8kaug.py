from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
from vllm.lora.request import LoRARequest
import json

llm = LLM(model="NousResearch/Llama-2-7b-hf", enable_lora=True, tensor_parallel_size=4)
lora_path = "ckpts/sft_gsm8kaug_llama7B_subsample/checkpoint_040000/hf_model/"

def prepare_sample(example):
    prompt = example["query"] + "\nAnswer: "
    response_orig = example["response"]
    answer_idx = response_orig.find("The answer is")
    if answer_idx>=0:
        response = response_orig[:answer_idx]
        answer = "#### "+response_orig[answer_idx+len("The answer is"):].replace(":", "").replace("$", "").replace(".", "").replace("\\boxed{", "").replace("}", "").strip()
        response += answer
    else:
        response = ""
    return (prompt, response)


with open('ckpts/AugGSM8K_part1.jsonl', 'r') as json_file:
    json_list = list(json_file)

with open('ckpts/AugGSM8K_part2.jsonl', 'r') as json_file:
    json_list += list(json_file)

train_examples = []
for json_str in json_list:
    result = json.loads(json_str)
    train_examples.append(result)

train_samples_orig = list(map(prepare_sample, train_examples))
eval_questions = []
eval_answers = []
for sample in train_samples_orig:
    if len(sample[1])>0:
        eval_questions.append(sample[0])
        eval_answers.append(sample[1])


eval_questions = np.array(eval_questions)
eval_answers = np.array(eval_answers)

sampling_params = SamplingParams(
    n = 4,
    temperature=0.8,
    max_tokens=512,
    top_p=0.95,
    seed = 1, 
)


# eval_questions = test_questions
# eval_answers = test_answers

output = llm.generate(eval_questions, sampling_params, lora_request=LoRARequest("lora_adapter", 1, lora_path))

def extract_latex(text):
    start = text.find("#### ") + len("#### ")
    return text[start:]
    
def answer_type_individial(output , answer):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
        
    answer = extract_latex(answer)
    output_answer_start_idx = output.find("#### ")
    
    if output_answer_start_idx != -1:
        output = output[output_answer_start_idx+len("#### "):]
        if output.replace(",", "") == answer.replace(",", ""):
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


answer_types_all = []
# answers_all = []
for i in range(len(output)):
    answer_types = []
    # answers = []
    for item in output[i].outputs:
        # answers.append(item.text)
        answer_type = answer_type_individial(item.text, eval_answers[i])
        answer_types.append(answer_type)
    answer_types_all.append(answer_types)
    # answers_all.append(answers)

answer_types_all = np.array(answer_types_all)
# np.save(lora_path+"train_answer_types_16.npy", answer_types_all)
np.save(lora_path+"train_answer_types_4_seed1.npy", answer_types_all)

# print((answer_types_all==0).mean(axis=-1).mean())
# import IPython; IPython.embed()