from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser



# pretrained_model_name_or_path = "../ckpts/sft_bios_new_llama7B_2/checkpoint_02000/hf_model"

# pretrained_model_name_or_path = "../ckpts/sft_gsm8k_llama7B_subsample1/checkpoint_10000/hf_model"
pretrained_model_name_or_path = "../ckpts/sft_gsm8k_llama7B_subsample2_easy75/checkpoint_02000/hf_model"
output_name  = pretrained_model_name_or_path+"_merged"
trained_adapter_config = PeftConfig.from_pretrained(pretrained_model_name_or_path)

peft_config = trained_adapter_config

# Use the pretrained (local or remote) peft adapter file "adapter_config.json"

base_model = AutoModelForCausalLM.from_pretrained(
    trained_adapter_config.base_model_name_or_path, return_dict=True, torch_dtype=torch.bfloat16
)

# Load the peft weights in "adapter_model.bin" and wrap the base model with a PeftModel
base_model2 = PeftModel.from_pretrained(
    base_model,
    pretrained_model_name_or_path)
base_model2.eval()


# import IPython; IPython.embed(); exit(1)

base_model2 = base_model2.merge_and_unload()
base_model2.generation_config.temperature=None
base_model2.do_sample = True
base_model2.generation_config.top_p=None

base_model2.save_pretrained(f"{output_name}")

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
tokenizer.save_pretrained(f"{output_name}")