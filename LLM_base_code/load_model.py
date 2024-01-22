import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
def load_model(model_path,load_in_8bit=True):
    device_map = 'auto'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        load_in_8bit=load_in_8bit,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            llm_int8_threshold=6.0
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path,  max_length=256, truncation=True, padding=True,trust_remote_code=True)
    return model, tokenizer
# if __name__ == "__main__":
#     model, tokenizer = load_model('')
#     print(model)