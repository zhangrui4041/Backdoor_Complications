import os
import transformers
import torch

def save_model(model_name, model, epoch, saved_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if not os.path.exists(f"{saved_path}/model_epoch_{str(epoch)}"):
        os.makedirs(f"{saved_path}/model_epoch_{str(epoch)}")
    model.config.to_json_file(f"{saved_path}/model_epoch_{str(epoch)}/config.json")
    torch.save(
        model.state_dict(),
        f"{saved_path}/model_epoch_{str(epoch)}/pytorch_model.bin",
    )
    tokenizer.save_pretrained(f"{saved_path}/model_epoch_{str(epoch)}/")