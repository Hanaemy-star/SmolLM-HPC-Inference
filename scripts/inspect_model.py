from transformers import AutoModelForCausalLM, AutoTokenizer


checkpoint = "HuggingFaceTB/SmolLM-135M"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
weights = model.state_dict()
#with open("model_weights.txt", "a") as f:
#    for layer_name, tensor in weights.items():
#        f.write(f"Слой: {layer_name} | Форма: {str(tensor.shape)} | Тип: {tensor.dtype}\n")
with open("layers_config.txt", "a") as conf:
    with open("model.bin", "wb") as mod:
        for layer_name, tensor in weights.items():
            conf.write(f"{tensor.shape}\n")
            array = tensor.cpu().float().numpy()
            mod.write(array.tobytes())

