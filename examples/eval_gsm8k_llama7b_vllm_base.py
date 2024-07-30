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
    n = 12,
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
    stop="Q:"
) 


# eval_questions = train_questions
# eval_questions = [question + "\nAnswer: " for question in eval_questions]

# hard
# q1 = "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck."
# a1 = 'If there were 26 pink hard hats and Carl took away 4 pink hard hats, the number of pink hard hats that remained is 26-4 = <<26-4=22>>22\nJohn also took away 6 pink hard hats, leaving 22-6 = <<22-6=16>>16 pink hard hats in the truck.\nIf John also took twice as many green hard hats as pink hard hats, he took 2*6 = <<6*2=12>>12 green hard hats.\nThe total number of green hard hats that remained in the truck is 15-12 = <<15-12=3>>3\nIn the truck, after some are taken, there were 3 green hard hats + 16 pink hard hats = <<3+16=19>>19 hard hats in the truck.\nAltogether, 19 green and pink hard hats + 24 yellow hards hats = <<19+24=43>>43 hard hats remained in the truck\n#### 43'
# q2 = "Nancy is filling an aquarium for her fish. She fills it halfway and goes to answer the door. While she's gone, her cat knocks the aquarium over and spills half the water in it. Then Nancy comes back and triples the amount of water in the aquarium. If the aquarium is 4 feet long, 6 feet wide, and 3 feet high, how many cubic feet of water are in the aquarium?"
# a2 = "First calculate the volume of the aquarium by multiplying its length, width and height: 4 ft * 6 ft * 3 ft = <<4*6*3=72>>72 cubic ft\nThen figure out what proportion of the aquarium is full after the cat knocks it over: 1/2 * 1/2 = 1/4\nThen figure out what proportion of the aquarium is full after Nancy refills it: 3 * 1/4 = 3/4\nNow multiply the proportion of the aquarium that's full by the aquarium's volume to find out how much water is in it: 72 cubic ft * 3/4 = <<72*3/4=54>>54 cubic ft\n#### 54"
# q3 = "Studying for her test, Mitchell had read ten chapters of a book before 4 o'clock. When it clocked 4, Mitchell had read 20 pages of the 11th chapter of the book she was studying from. After 4 o'clock, she didn't read the remaining pages of chapter eleven but proceeded and read 2 more chapters of the book. If each chapter in the book had 40 pages, calculate the total number of pages that Mitchell had read altogether?"
# a3 = 'Since each chapter of the book has 40 pages, Mitchell had read 10*40 = <<10*40=400>>400 pages from the first ten chapters.\nAfter reading 20 pages of the eleventh chapter, the total number of pages that Mitchell had read is 400+20 = <<400+20=420>>420\nThe next two chapters that she read had 2*40 = <<2*40=80>>80 pages.\nIn total, Mitchell read 420+80 = <<420+80=500>>500 pages of the book that day.\n#### 500'


# easy
q1 = 'Santino has 2 papaya trees and 3 mango trees. If each papaya tree produces 10 papayas and each mango tree produces 20 mangos, how many fruits does Santino have in total?'
a1 = 'Santino has 2 * 10 = <<2*10=20>>20 papaya fruits\nSantino has 3 * 20 = <<3*20=60>>60 mango fruits\nIn total, Santino has 20 + 60 = <<20+60=80>>80 fruits\n#### 80'
q2 = 'Brandon has a collection of 20 baseball cards.  Malcom has 8 more cards than Brandon.  However, then Malcom gives half of his cards to his friend Mark.  How many cards does Malcom have left?'
a2 = 'Malcom has 20 cards + 8 cards = <<20+8=28>>28 cards.\nMalcom gives away 1/2 * 28 cards = <<1/2*28=14>>14 cards to Mark.\nMalcom has 28-14 cards = <<28-14=14>>14 cards remaining.\n#### 14'
q3 = 'Arwen and Elrond picked some flowers. Arwen was able to get 20 tulips and Elrond was able to get twice as many tulips as Arwen did. How many tulips were they able to get in total?'
a3 = 'Elrond was able to get 20 x 2 = <<20*2=40>>40 tulips.\nTherefore, the total number of tulips they were able to get is 40 + 20 = <<40+20=60>>60.\n#### 60'

def get_eval_prompt(orig_questions, orig_answers):
    prompts_all = []
    for i in range(len(orig_questions)):
#         prompt = f"Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. #### 29\n\n\
# Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33\n\n\
# Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. #### 8\n\n\
# Q: {orig_questions[i]}\nA:"
        prompt = f"Q: {q1}\nA: {a1}\n\nQ: {q2}\nA: {a2}\n\nQ: {q3}\nA: {a3}\n\nQ: {orig_questions[i]}\nA:"
        # for j in range(1, 5):
        #     prompt += "Q: "+ orig_questions[(i+j)%len(orig_questions)] + "\n"
        #     prompt += "A: "+ orig_answers[(i+j)%len(orig_answers)] + "\n\n"
        # prompt+="Q: "+orig_questions[i] + "\nA: "
        prompts_all.append(prompt)
    return prompts_all
        
# eval_questions = get_eval_prompt(train_questions, train_answers)
# eval_answers = train_answers

eval_questions = get_eval_prompt(test_questions, test_answers)
eval_answers = test_answers


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
            output=output[:output_answer_end_idx]
        output = output.replace(",", "")
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
    answers_all.append(answers)


answer_types_all = np.array(answer_types_all)
np.save("llama7B_GSM8k_test_answer_types_all12_easy_context.npy", answer_types_all)

import IPython; IPython.embed()