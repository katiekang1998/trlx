from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
from vllm.lora.request import LoRARequest

# llm = LLM(model="ckpts/sft_gsm8k_llama7B_subsample1/checkpoint_10000/hf_model_merged/", tensor_parallel_size=2)  # Name or path of your model

llm = LLM(model="NousResearch/Llama-2-7b-hf", tensor_parallel_size=2)
# output = llm.generate("Hello, my name is")
# print(output)




titles = np.load("movies/test_titles.npy")

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
        prompt = f"Title: The Hobbit\nPremise: Gandalf tricks Bilbo into hosting a party for Thorin and his band of dwarves, who sing of reclaiming the Lonely Mountain and its vast treasure from the dragon Smaug.\n\n\
            Title: Siva (1989 Telugu film)\nPremise: Siva, a student, comes to Vijayawada from a nearby town to pursue his education.\n\n\
            Title: The Godfather\nPremise: In 1945, at his daughter Connie's wedding, Vito Corleone hears requests in his role as the Godfather, the Don of a New York crime family.\n\n\
            Title: Great Expectations\nPremise: On Christmas Eve, around 1812, Pip, an orphan who is about seven years old, encounters an escaped convict in the village churchyard, while visiting the graves of his parents and siblings.\n\n\
            Title: {names[i]}\nPremise:"
        prompts_all.append(prompt)
    return prompts_all
        
eval_questions = get_eval_prompt(titles)

output = llm.generate(eval_questions, sampling_params)



responses_all = []
for i in range(len(output)):
    responses = []
    for item in output[i].outputs:
        responses.append(item.text)
    responses_all.append(responses)
    

responses_all = np.array(responses_all)
np.save("plots_base_model_samples.npy", responses_all)