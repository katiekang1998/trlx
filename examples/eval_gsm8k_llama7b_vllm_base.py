from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
from vllm.lora.request import LoRARequest

# llm = LLM(model="ckpts/sft_gsm8k_llama7B_subsample1/checkpoint_10000/hf_model_merged/", tensor_parallel_size=2)  # Name or path of your model

llm = LLM(model="NousResearch/Llama-2-7b-hf", tensor_parallel_size=4)
# output = llm.generate("Hello, my name is")
# print(output)


dataset = load_dataset("gsm8k", "main")
train_questions = dataset["train"]["question"]
train_answers = dataset["train"]['answer']

test_questions = dataset["test"]["question"]
test_answers = dataset["test"]['answer']


sampling_params = SamplingParams(
    n = 100,
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)


# eval_questions = train_questions
# eval_questions = [question + "\nAnswer: " for question in eval_questions]


def get_eval_prompt(orig_questions, orig_answers):
    prompts_all = []
    for i in range(len(orig_questions)):
        prompt = f"Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.\n#### 29\n\n\
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33\n\n\
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.\n#### 8\n\n\
Q: {orig_questions[i]}\nA: "
        # for j in range(1, 5):
        #     prompt += "Q: "+ orig_questions[(i+j)%len(orig_questions)] + "\n"
        #     prompt += "A: "+ orig_answers[(i+j)%len(orig_answers)] + "\n\n"
        # prompt+="Q: "+orig_questions[i] + "\nA: "
        prompts_all.append(prompt)
    return prompts_all
        
eval_questions = get_eval_prompt(train_questions, train_answers)
eval_answers = train_answers

output = llm.generate(eval_questions, sampling_params)

def extract_latex(text):
    start = text.find("#### ") + len("#### ")
    return text[start:].replace(",", "")
    
def answer_type_individial(output , answer):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
        
    answer = extract_latex(answer)
    output_answer_start_idx = output.find("#### ")
    output_answer_end_idx = output[output_answer_start_idx+len("#### "):].find("\n")
    
    if output_answer_start_idx != -1:
        output = output[output_answer_start_idx+len("#### "):]
        if output_answer_end_idx!=-1:
            output=output[:output_answer_end_idx].replace(",", "")
        if output == answer:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type, output


answer_types_all = []
answers_all = []
for i in range(len(output)):
    answer_types = []
    answers = []
    for item in output[i].outputs:
        answer_type, answer = answer_type_individial(item.text, eval_answers[i])
        answer_types.append(answer_type)
        # answer_types.append(item.text)
        answers.append(answer)
    answer_types_all.append(answer_types)
    # answers_all.append(answers)

# import IPython; IPython.embed(); exit()

answer_types_all = np.array(answer_types_all)
np.save("llama7B_GSM8k_train_answer_types_all100.npy", answer_types_all)