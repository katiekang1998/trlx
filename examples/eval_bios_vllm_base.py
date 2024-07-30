from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
from vllm.lora.request import LoRARequest

# llm = LLM(model="ckpts/sft_gsm8k_llama7B_subsample1/checkpoint_10000/hf_model_merged/", tensor_parallel_size=2)  # Name or path of your model

llm = LLM(model="NousResearch/Llama-2-7b-hf", tensor_parallel_size=2)
# output = llm.generate("Hello, my name is")
# print(output)




names = np.load("biographies/test_names.npy")

sampling_params = SamplingParams(
    n = 2,
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
    stop="\n"
)

def get_eval_prompt(names):
    prompts_all = []
    for i in range(len(names)):
        prompt = f"Name: Emmett Till\nBio: Emmett Till was an African American boy who was abducted, tortured, and lynched in Mississippi in 1955 at the age of 14, after being accused of offending a white woman, Carolyn Bryant, in her family's grocery store.\n\n\
            Name: Nikki Haley\nBio: Nikki Haley is an American politician and diplomat who served as Governor of South Carolina from 2011 to 2017, and as the 29th United States ambassador to the United Nations from January 2017 through December 2018.\n\n\
            Name: Guillermo Saavedra (footballer)\nBio: Guillermo Saavedra was a Chilean football midfielder, who played for his native country in the 1930 FIFA World Cup.\n\n\
            Name: Bill Murray\nBio: Bill Murray is an American actor and comedian, known for his deadpan delivery in roles ranging from studio comedies to independent dramas.\n\n\
            Name: {names[i]}\nBio:"
        prompts_all.append(prompt)
    return prompts_all
        
eval_questions = get_eval_prompt(names)

output = llm.generate(eval_questions, sampling_params)



responses_all = []
for i in range(len(output)):
    responses = []
    for item in output[i].outputs:
        responses.append(item.text)
    responses_all.append(responses)
    


responses_all = np.array(responses_all)
np.save("bios_base_model_samples.npy", responses_all)