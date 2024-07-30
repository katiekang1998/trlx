from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
from vllm.lora.request import LoRARequest
import re
import string

llm = LLM(model="NousResearch/Llama-2-7b-hf", enable_lora=True, tensor_parallel_size=4)
# lora_path = "ckpts/sft_gsm8k_llama7B_full3/checkpoint_10000/hf_model/"



dataset = load_dataset("trivia_qa", "rc.nocontext")

train_questions = dataset["train"]["question"]
train_answers = dataset["train"]['answer']

test_questions = dataset["validation"]["question"]
test_answers = dataset["validation"]['answer']


sampling_params = SamplingParams(
    n = 12,
    temperature=0.8,
    max_tokens=50,
    top_p=0.95,
    stop="\n"
)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

shots = '''Q: In which country is the Aswan Dam?
A: egypt

Q: The boll weevil, a species of beetle, causes damage to which crop?
A: cotton

Q: Nov 30, 1835 saw the birth of what famed American humorist and novelist, known for works such as The Prince and the Pauper and A Connecticut Yankee in King Arthur's Court, along with some other famous works?
A: mark twain

Q: President George Washington sent the proposed first ten amendments to the US Constitution to the senate for ratification on October 2, 1789. By what name are these amendments commonly known?
A: bill of rights

Q: Anchored by the star Antares, the constellation Scorpius represents what animal?
A: scorpion

'''

eval_questions = test_questions
eval_questions = [shots + "Q: "+ question + "\nA:" for question in eval_questions]
eval_answers = test_answers

# output = llm.generate(eval_questions, sampling_params, lora_request=LoRARequest("lora_adapter", 1, lora_path))
output = llm.generate(eval_questions, sampling_params)


def answer_type_individial(output , answer_dict):
    
    answer = normalize_answer(answer_dict["value"])
    aliases = [normalize_answer(alias) for alias in answer_dict["aliases"]]

    output = output[1:]
    if "\n" in output:
        output = (output[output.find("\n"):])
    output = normalize_answer(output)


    aliases_normalized = []
    for alias in aliases:
        aliases_normalized.append((alias))

    if output == (answer) or output in aliases_normalized:
        answer_type = 0
    else:
        answer_type = 1
    return answer_type, output


answer_types_all = []
# outputs_all = []
for i in range(len(output)):
    answer_types = []
    # outputs = []
    for item in output[i].outputs:
        # print(item.text)
        answer_type, filtered_output = answer_type_individial(item.text, eval_answers[i])
        answer_types.append(answer_type)
        # outputs.append(filtered_output)
    answer_types_all.append(answer_types)
    # outputs_all.append(outputs)

answer_types_all = np.array(answer_types_all)
np.save("llama7B_triviaqa_test_answer_types_all12.npy", answer_types_all)