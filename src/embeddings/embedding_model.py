import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2"
)
model = AutoModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2")


def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings
