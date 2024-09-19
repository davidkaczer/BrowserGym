from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
from transformers.image_utils import load_image

image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZA-l1mZfUXDUT7aNbc0LBOgflUZOpmJvdng&s"

model_name = "HuggingFaceM4/Idefics3-8B-Llama3"

pipe = pipeline(
    # task="text-generation",
    task="image-to-text",
    model=model_name,
    device_map="auto",
    max_new_tokens=2000,
    model_kwargs={},
)
llm = HuggingFacePipeline(pipeline=pipe)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages_formated = {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"},
        ]
    },

prompt = tokenizer.apply_chat_template(messages_formated, tokenize=False)

print(prompt)

response = llm.invoke(prompt)

print(response)