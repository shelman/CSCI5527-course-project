from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
print(tokenizer("24")["input_ids"])


print(tokenizer("25")["input_ids"])
