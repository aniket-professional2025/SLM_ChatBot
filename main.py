# Importing Required Libraries
from transformers import LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
from dotenv import load_dotenv

# Loading Environmental Variables
load_dotenv()

# Setting the configurations
base_model_id = "meta-llama/Llama-2-7b-chat-hf"
adapter_path = r"C:\Users\Webbies\Jupyter_Notebooks\SLMs\finetunedModel\checkpoint-20"
print("----------------- Based Model and Adapter set succesfully --------------------")

# Setting Up the BitsAndBytes Configuration
nf4Config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
print("----------------- Bits and Bytes Configuration are Set Succesfully -----------------------")

# Loading the Based Model
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config = nf4Config, trust_remote_code = True, use_auth_token = True, device_map = "auto")
print("----------------- Load the Base Model ------------------------")

# Setting the Tokenizer and the fine tuned model
tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast = False, trust_remote_code = True, add_eos_token = True)
finemodel = PeftModel.from_pretrained(base_model, adapter_path)
print("------------------------ Setting the Tokenizer and The Fine Tunned Model ---------------------------")

# Load the model in Evaluation Mode
finemodel.eval()

# Function to Get model response
def get_answer(user_prompt):
    eval_prompt = f"Question:{user_prompt} Just answer this question accurately and concisely.\nAnswer:"
    tokens = tokenizer(eval_prompt, return_tensors = 'pt').to("cuda")
    with torch.no_grad():
        response = finemodel.generate(**tokens, max_new_tokens=100, do_sample = True, temperature = 0.7, top_p = 0.9, eos_token_id = tokenizer.eos_token_id)
        output = tokenizer.decode(response[0], skip_special_tokens = True)
        torch.cuda.empty_cache()

        if "Answer:" in output:
            final_output = output.split("Answer:")[1].strip()
        else:
            final_output = output.strip()
        
        final_output = final_output.replace(":", "").replace(";", "").strip()

        return final_output