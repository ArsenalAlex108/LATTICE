import torch
from transformers import AutoModel, AutoTokenizer

model_name = "sentence-transformers/msmarco-distilbert-base-tas-b"
model = AutoModel.from_pretrained(model_name)
model.eval()  # set to evaluation mode
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare a dummy input: adjust shape (batch_size, sequence_length) as needed.
# Here, we assume a sequence length of 128.
dummy_text = "How many people live in London?"
encoded = tokenizer(dummy_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# Export with opset version 14 (or higher) to support scaled dot product attention.
torch.onnx.export(
    model,
    args=(encoded["input_ids"], encoded.get("attention_mask", None)),  # include other inputs if needed
    f="msmarco-distilbert-base-tas-b.onnx",
    opset_version=14,
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size"}
    }
)