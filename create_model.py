from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

def download_models(base_dir: str):
    # 1. Download the Oryx-7B model (multimodal language model)
    oryx7b_dir = f"{base_dir}/THUdyh-Oryx-7b"
    snapshot_download(
        repo_id="THUdyh/Oryx-7B",
        local_dir=oryx7b_dir,
        local_dir_use_symlinks=False
    )
    print(f"✅ Oryx‑7B model downloaded to: {oryx7b_dir}")

    # 2. Download the Oryx‑ViT vision encoder
    oryxvit_dir = f"{base_dir}/THUdyh-Oryx-ViT"
    snapshot_download(
        repo_id="THUdyh/Oryx-ViT",
        local_dir=oryxvit_dir,
        local_dir_use_symlinks=False
    )
    print(f"✅ Oryx‑ViT model downloaded to: {oryxvit_dir}")

    return oryx7b_dir, oryxvit_dir

def load_models(oryx7b_dir: str, oryxvit_dir: str):
    # Load the multimodal LLM
    from oryx.model.builder import load_pretrained_model
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=oryx7b_dir,
        model_base=None,
        model_name="oryx_qwen",
        device_map="auto"
    )
    print("🔄 Loaded Oryx‑7B (tokenizer, model, image_processor)")

    # For direct PyTorch load of ViT if needed
    import torch
    from torch import nn

    vit_weights = torch.load(f"{oryxvit_dir}/oryx_vit.pth", map_location="cpu")
    # Assume SigLip architecture – you’d define the model class or import it
    from oryx.mm_utils import SigLipVisionEncoder  # hypothetical path
    vit_model = SigLipVisionEncoder()
    vit_model.load_state_dict(vit_weights)
    vit_model.eval()
    print("🔄 Loaded Oryx‑ViT vision encoder")

    return tokenizer, model, image_processor, vit_model

if __name__ == "__main__":
    base_dir = "./models"
    oryx7b_dir, oryxvit_dir = download_models(base_dir)
    tokenizer, model, image_processor, vit_model = load_models(oryx7b_dir, oryxvit_dir)
