from transformers import LlamaTokenizer
base_model = "/mnt/llms/model/meta-llama/Llama-2-7b-hf"
prompt = "I believe the meaning of life is"
tokenizer = LlamaTokenizer.from_pretrained(base_model)
input_ids = tokenizer(prompt, return_tensors="pt")
print(input_ids)