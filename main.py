import os
import json
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

# === Define the Autoencoder ===
class BottleneckT5Autoencoder:
    def __init__(self, model_path: str, device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.model.eval()
        self.last_bottleneck = None

    @torch.no_grad()
    def embed(self, text: str) -> torch.FloatTensor:
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        decoder_inputs = self.tokenizer('', return_tensors='pt').to(self.device)
        bottleneck = self.model(
            **inputs,
            decoder_input_ids=decoder_inputs['input_ids'],
            encode_only=True,
        )[0]
        self.last_bottleneck = bottleneck.squeeze(0).detach()
        return self.last_bottleneck

# === Step 1: Load Dialogues from TXT ===
def load_dialogues_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    dialogues = []
    for line in lines:
        utterances = [utt.strip() for utt in line.strip().split('__eou__') if utt.strip()]
        if len(utterances) >= 2:
            dialogues.append(utterances)
    return dialogues

# === Step 2: Embed and Save ===
def embed_and_save(dialogues, autoencoder, save_path="utterance_embeddings2.json", limit=None):
    all_embeddings = []
    for dialog in tqdm(dialogues[:limit] if limit else dialogues, desc="🔄 Embedding utterances"):
        emb_dialog = []
        for utt in dialog:
            try:
                emb = autoencoder.embed(utt).cpu().tolist()
                emb_dialog.append({
                    "utterance": utt,
                    "embedding": emb
                })
            except Exception as e:
                print(f"⚠️ Skipping utterance due to error: {e}")
        all_embeddings.append(emb_dialog)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_embeddings, f, indent=2)
    print(f"✅ Saved to {save_path}")

# === Run All ===
if __name__ == "__main__":
    txt_file_path = "train/output.txt"  # adjust if needed

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    autoencoder = BottleneckT5Autoencoder('thesephist/contra-bottleneck-t5-base-wikipedia', device=device)

    print("📄 Loading dialogues...")
    dialogues = load_dialogues_from_txt(txt_file_path)
    print(f"📚 Found {len(dialogues)} dialogues.")

    embed_and_save(dialogues, autoencoder, save_path="train/dialogues_train2.json")  # adjust limit if needed
