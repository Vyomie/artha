import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Liquid Network ---
class LiquidLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(output_dim, input_dim) * 0.02)
        self.U = nn.Parameter(torch.randn(output_dim, output_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.act = nn.Tanh()

    def forward(self, x, prev_state=None):
        if prev_state is None:
            prev_state = torch.zeros(x.size(0), self.W.size(0), device=x.device)
        return self.act(x @ self.W.T + prev_state @ self.U.T + self.bias)

class LiquidNetwork(nn.Module):
    def __init__(self, in_dim=768, h_dim=4000, out_dim=768):
        super().__init__()
        self.l1 = LiquidLayer(in_dim, h_dim)
        self.l2 = LiquidLayer(h_dim, h_dim)
        self.l3 = LiquidLayer(h_dim, h_dim)
        self.l4 = LiquidLayer(h_dim, h_dim)
        self.l5 = nn.Linear(h_dim * 4, out_dim)

    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        h4 = self.l4(h3)
        return self.l5(torch.cat([h1, h2, h3, h4], dim=-1))

# --- Bottleneck Autoencoder ---
class BottleneckT5Autoencoder:
    def __init__(self, model_path='thesephist/contra-bottleneck-t5-base-wikipedia', device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, text: str):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        decoder_inputs = self.tokenizer('', return_tensors='pt').to(self.device)
        return self.model(
            **inputs,
            decoder_input_ids=decoder_inputs['input_ids'],
            encode_only=True
        )[0].squeeze(0).detach()

    @torch.no_grad()
    def generate_from_latent(self, latent, max_length=512, temperature=1.0):
        dummy_text = '.'
        dummy = self.embed(dummy_text)
        perturb_vector = latent - dummy
        self.model.perturb_vector = perturb_vector
        input_ids = self.tokenizer(dummy_text, return_tensors='pt').to(self.device).input_ids
        output = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=0.9,
            temperature=temperature
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# --- Plug-and-play Pipeline ---
class Pipeline:
    def __init__(self, model_name: str, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.autoencoder = BottleneckT5Autoencoder(device=self.device)
        self.model = LiquidNetwork().to(self.device)

        state_dict = torch.hub.load_state_dict_from_url(
            f"https://huggingface.co/{model_name}/resolve/main/model.pth",
            map_location=self.device
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def __call__(self, prompt: str) -> str:
        with torch.no_grad():
            latent = self.model(self.autoencoder.embed(prompt).unsqueeze(0).to(self.device))
            return self.autoencoder.generate_from_latent(latent.squeeze(0))
