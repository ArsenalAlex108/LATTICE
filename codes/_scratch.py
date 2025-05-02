# %%
from huggingface_hub import snapshot_download

snapshot_download( \
    "sentence-transformers/msmarco-distilbert-base-tas-b",
    local_dir="D:/Projects/CSharp/ECAIService/ECAIService/Resources/Models/sentence-transformers/msmarco-distilbert-base-tas-b",
    allow_patterns=["pytorch_model.bin", "config.json", "tokenizer.json", "vocab.txt"],
    ignore_patterns=["*.py", "*.md", "*.txt", "*.sh", "flax_model.msgpack", "tf_model.h5"]
)

# %%


# %%
