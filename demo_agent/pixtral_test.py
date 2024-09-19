import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
import json
from PIL import Image
import numpy as np
from huggingface_hub import snapshot_download

from mistral_common.protocol.instruct.messages import (
    UserMessage,
    TextChunk,
    ImageURLChunk,
    ImageChunk,
)
from PIL import Image
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

class GELU(nn.Module):
    def __init__(self, dim_in, dim_out, approximate='none', bias=True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, x):
        if self.approximate == 'tanh':
            return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
        else:
            return F.gelu(self.linear(x))

class Rope2D(nn.Module):
    def __init__(self, dim, max_position_embeddings=1024, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Conv2d(config['num_channels'], config['hidden_size'], kernel_size=config['patch_size'], stride=config['patch_size'])
        self.rope = Rope2D(config['hidden_size'] // config['num_attention_heads'], base=config['rope_theta'])
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=config['hidden_size'], nhead=config['num_attention_heads'], dim_feedforward=config['intermediate_size']) for _ in range(config['num_hidden_layers'])])
        self.norm = nn.LayerNorm(config['hidden_size'])
        self.gelu = GELU(config['hidden_size'], config['hidden_size'])

    def forward(self, pixel_values):
        x = self.embed(pixel_values)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        cos, sin = self.rope(x, seq_len=h*w)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.gelu(x)
        return x

class TextDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config['vocab_size'], config['dim'])
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model=config['dim'], nhead=config['n_heads'], dim_feedforward=config['hidden_dim']) for _ in range(config['n_layers'])])
        self.norm = nn.LayerNorm(config['dim'])
        self.lm_head = nn.Linear(config['dim'], config['vocab_size'], bias=False)

    def forward(self, input_ids, encoder_hidden_states):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, encoder_hidden_states)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

class PixtralModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.vision_encoder = VisionEncoder(params['vision_encoder'])
        self.text_decoder = TextDecoder(params)

    def forward(self, image, input_ids):
        vision_output = self.vision_encoder(image)
        logits = self.text_decoder(input_ids, vision_output)
        return logits

def load_model(params, model_path):
    model = PixtralModel(params)
    
    with safe_open(f'{model_path}/consolidated.safetensors', framework="pt", device="cpu") as f:
        for name, param in model.named_parameters():
            if name in f.keys():
                param.data = f.get_tensor(name)
    
    model.eval()
    model.cuda() 
    return model

def process_image(image):
    image = image.convert('RGB')
    image = image.resize((params['vision_encoder']['image_size'], params['vision_encoder']['image_size']))
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return image_tensor.cuda()

def generate_text(vision_output, max_length=100):
    input_ids = torch.tensor([[tokenizer.instruct_tokenizer.tokenizer.bos_id]]).cuda()
    
    for _ in range(max_length):
        with torch.no_grad():
            logits = model.text_decoder(input_ids, vision_output)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        if next_token.item() == tokenizer.instruct_tokenizer.tokenizer.eos_id:
            break
    
    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


model_path = "/home/ubuntu/pixtral"

with open(f'{model_path}/params.json', 'r') as f:
    params = json.load(f)

with open(f'{model_path}/tekken.json', 'r') as f:
    tokenizer_config = json.load(f)

tokenizer = MistralTokenizer.from_model("pixtral")
model = load_model(params=params, model_path=model_path)

image = Image.new('RGB', (64, 64))
image_tensor = process_image(image)

# tokenize images and text
tokenized = tokenizer.encode_chat_completion(
    ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="Describe this image"),
                    ImageChunk(image=image),
                ]
            )
        ],
        model="pixtral",
    )
)
tokens, text, images = tokenized.tokens, tokenized.text, tokenized.images

# Count the number of tokens
print("# tokens", len(tokens))
print("# images", len(images))

with torch.no_grad():
    vision_output = model.vision_encoder(image_tensor)
    generated_text = generate_text(vision_output)
    
    