from transformers import pipeline, AutoProcessor
from PIL import Image    
import requests

model_id = "llava-hf/llama3-llava-next-8b-hf"
# pipe = pipeline("image-to-text", model=model_id, device="cuda:0")
pipe = pipeline("text-generation", model=model_id, device="cuda:0")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"},
          {"type": "image"},
        ],
    },
]
processor = AutoProcessor.from_pretrained(model_id)
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
outputs = pipe(text_inputs=prompt, max_new_tokens=200)
print(outputs)

{"generated_text": "\nUSER: What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud\nASSISTANT: Lava"}