from datasets import load_dataset
from transformers import AutoTokenizer
import json

def load_alpaca_dataset(data_file, tokenizer, max_len):
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors=None,
        )
        
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < max_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if add_eos_token and len(result["input_ids"]) >= max_len:
            result["input_ids"][max_len - 1] = tokenizer.eos_token_id
            result["attention_mask"][max_len - 1] = 1

        result["labels"] = result["input_ids"].copy()
        return result


    # def generate_and_tokenize_prompt(data_point):
    #     full_prompt = ""
    #     instruction = data_point['instruction']
    #     output = data_point['output']
    #     if data_point['input'] and data_point['input']!= '':
    #         input = data_point['input']
    #         full_prompt = [
    #             {"role": "system", "content": input},
    #             {"role": "user", "content": instruction},
    #             {"role": "assistant", "content": output}
    #         ]
    #     else:
    #         full_prompt = [
    #             {"role": "user", "content": instruction},
    #             {"role": "assistant", "content": output}
    #         ]
    #     full_prompt = tokenizer.bos_token+json.dumps(full_prompt)+tokenizer.eos_token
    #     tokenized_full_prompt = tokenize(full_prompt)
    #     return tokenized_full_prompt
    def generate_prompt(instruction,output):
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:
    {output}
    """

    def generate_model_prompt(data_point):
        full_prompt = ""
        instruction = data_point["instruction"]
        output = data_point["output"]
        full_prompt = generate_prompt(instruction,output)
        inputs = tokenize(full_prompt)
        return inputs
    data = load_dataset("json", data_files=data_file)["train"]
    data = data.map(generate_model_prompt, num_proc=8)
    
    return data

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/data/wpd/Wizardcoder-3B", trust_remote_code=True)
    ds = load_alpaca_dataset("./data/last_method/function/function_benchmark_alpaca.jsonl", tokenizer, 512)
    print(ds[1])
    print(tokenizer.decode(ds[1]['input_ids']))