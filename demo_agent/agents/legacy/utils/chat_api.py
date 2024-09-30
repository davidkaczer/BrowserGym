from dataclasses import asdict, dataclass
import io
import json
from .prompt_templates import PromptTemplate, get_prompt_template
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from functools import partial
from typing import Optional, List, Any
import logging
from typing import Tuple
import time
import torch
import base64
import PIL
import openai
from copy import deepcopy

from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage
from langchain.chat_models.base import SimpleChatModel
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
from transformers import pipeline
from transformers.image_utils import load_image
from dataclasses import dataclass
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
from transformers import GPT2TokenizerFast


@dataclass
class ChatModelArgs:
    """Serializable object for instantiating a generic chat model.

    Attributes
    ----------
    model_name : str
        The name or path of the model to use.
    model_url : str, optional
        The url of the model to use, e.g. via TGI. If None, then model_name or model_path must
        be specified.
    eai_token: str, optional
        The EAI token to use for authentication on Toolkit. Defaults to snow.optimass_account.cl4code's token.
    temperature : float
        The temperature to use for the model.
    max_new_tokens : int
        The maximum number of tokens to generate.
    hf_hosted : bool
        Whether the model is hosted on HuggingFace Hub. Defaults to False.
    info : dict, optional
        Any other information about how the model was finetuned.
    DGX related args
    n_gpus : int
        The number of GPUs to use. Defaults to 1.
    tgi_image : str
        The TGI image to use. Defaults to "e3cbr6awpnoq/research/text-generation-inference:1.1.0".
    ace : str
        The ACE to use. Defaults to "servicenow-scus-ace".
    workspace : str
        The workspace to use. Defaults to UI_COPILOT_SCUS_WORKSPACE.
    max_total_tokens : int
        The maximum number of total tokens (input + output). Defaults to 4096.
    """

    model_name: str = "openai/gpt-3.5-turbo"
    model_url: str = None
    temperature: float = 0.1
    max_new_tokens: int = None
    max_total_tokens: int = None
    max_input_tokens: int = None
    hf_hosted: bool = False
    info: dict = None
    n_retry_server: int = 4

    def __post_init__(self):
        if self.model_url is not None and self.hf_hosted:
            raise ValueError("model_url cannot be specified when hf_hosted is True")

    def make_chat_model(self):
        if self.model_name.startswith("openai"):
            _, model_name = self.model_name.split("/")
            return ChatOpenAI(
                model_name=model_name,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
        else:
            try:
                chat = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    base_url="http://localhost:8000/v1",
                    )
                chat.invoke("")
                return chat
            except (openai.NotFoundError, openai.BadRequestError, openai.APIConnectionError) as e:
                return HuggingFaceChatModel(
                    model_name=self.model_name,
                    hf_hosted=self.hf_hosted,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    max_total_tokens=self.max_total_tokens,
                    max_input_tokens=self.max_input_tokens,
                    model_url=self.model_url,
                    n_retry_server=self.n_retry_server,
                )

    @property
    def model_short_name(self):
        if "/" in self.model_name:
            return self.model_name.split("/")[1]
        else:
            return self.model_name

    def key(self):
        """Return a unique key for these arguments."""
        return json.dumps(asdict(self), sort_keys=True)

    def has_vision(self):
        # TODO make sure to upgrade this as we add more models
        name_patterns_with_vision = [
            "vision",
            "4o",
            "llava",
            "Idefics",
            "Pixtral",
        ]
        return any(pattern in self.model_name for pattern in name_patterns_with_vision)


class HuggingFaceChatModel(SimpleChatModel):
    """
    Custom LLM Chatbot that can interface with HuggingFace models.

    This class allows for the creation of a custom chatbot using models hosted
    on HuggingFace Hub or a local checkpoint. It provides flexibility in defining
    the temperature for response sampling and the maximum number of new tokens
    in the response.

    Attributes:
        llm (Any): The HuggingFaceHub model instance.
        prompt_template (Any): Template for the prompt to be used for the model's input sequence.
    """

    llm: Any = Field(description="The HuggingFaceHub model instance")
    tokenizer: Any = Field(
        default=None,
        description="The tokenizer to use for the model",
    )
    prompt_template: Optional[PromptTemplate] = Field(
        default=None,
        description="Template for the prompt to be used for the model's input sequence",
    )
    n_retry_server: int = Field(
        default=4,
        description="The number of times to retry the server if it fails to respond",
    )

    def __init__(
        self,
        model_name: str,
        hf_hosted: bool,
        temperature: float,
        max_new_tokens: int,
        max_total_tokens: int,
        max_input_tokens: int,
        model_url: str = None,
        eai_token: str = None,
        n_retry_server: int = 1,
    ):
        """
        Initializes the CustomLLMChatbot with the specified configurations.

        Args:
            model_name (str): The path to the model checkpoint.
            prompt_template (PromptTemplate, optional): A string template for structuring the prompt.
            hf_hosted (bool, optional): Whether the model is hosted on HuggingFace Hub. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
            max_new_tokens (int, optional): Maximum length for the response. Defaults to 64.
            model_url (str, optional): The url of the model to use. If None, then model_name or model_name will be used. Defaults to None.
        """
        super().__init__()

        self.n_retry_server = n_retry_server

        if max_new_tokens is None:
            max_new_tokens = max_total_tokens - max_input_tokens
            logging.warning(
                f"max_new_tokens is not specified. Setting it to {max_new_tokens} (max_total_tokens - max_input_tokens)."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if isinstance(self.tokenizer, GPT2TokenizerFast):
            # TODO: make this less hacky once tokenizer.apply_chat_template is more mature
            logging.warning(
                f"No chat template is defined for {model_name}. Resolving to the hard-coded templates."
            )
            self.tokenizer = None
            self.prompt_template = get_prompt_template(model_name)

        if temperature < 1e-3:
            logging.warning(
                "some weird things might happen when temperature is too low for some models."
            )

        model_kwargs = {
            "temperature": temperature,
            "attn_implementation": "flash_attention_2",
        }

        self.__dict__["idefics_hack"] = False
        if model_url is not None:
            logging.info("Loading the LLM from a URL")
            client = InferenceClient(model=model_url, token=eai_token)
            self.llm = partial(
                client.text_generation, temperature=temperature, max_new_tokens=max_new_tokens
            )
        elif hf_hosted:
            logging.info("Serving the LLM on HuggingFace Hub")
            model_kwargs["max_length"] = max_new_tokens
            self.llm = HuggingFaceHub(repo_id=model_name, model_kwargs=model_kwargs)
        elif "Idefics" in model_name or "pixtral" in model_name:
            logging.info("Loading the LLM locally")
            self.__dict__["processor"] = AutoProcessor.from_pretrained(model_name)
            self.llm = AutoModelForVision2Seq.from_pretrained(
                model_name,
                # torch_dtype=torch.bfloat16,
                device_map="auto",
                )
            self.__dict__["max_new_tokens"] = max_new_tokens
            self.__dict__["idefics_hack"] = True
        else:
            logging.info("Loading the LLM locally")
            pipe = pipeline(
                task="text-generation",
                # task="image-to-text",
                model=model_name,
                device_map="auto",
                max_new_tokens=max_new_tokens,
                model_kwargs=model_kwargs,
                torch_dtype=torch.bfloat16,
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None or run_manager is not None or kwargs:
            logging.warning(
                "The `stop`, `run_manager`, and `kwargs` arguments are ignored in this implementation."
            )

        if self.idefics_hack:
            most_recent_image_index = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage) and type(message.content) is list][-1]

            image_b64 = messages[1].content[most_recent_image_index]['image_url']['url'].split(",")[1]
            image_pil = load_image(image_b64)
            
            messages_copy = deepcopy(messages)
            for i, message in enumerate(messages_copy):
                if isinstance(message, HumanMessage) and type(message.content) is list:
                    if i == most_recent_image_index:
                        message.content[1] = {"type": "image"}            
                    else:
                        message.content.pop()
                                
            messages_formated = _convert_messages_to_dict(messages_copy)
            for message in messages_formated:
                if isinstance(message["content"], str):
                    message["content"] = [{"type": "text", "text": message["content"]}]
                    
            prompt = self.processor.apply_chat_template(messages_formated, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image_pil], return_tensors="pt")
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
        elif self.tokenizer:
            messages_formated = _convert_messages_to_dict(messages)
            prompt = self.tokenizer.apply_chat_template(messages_formated, tokenize=False)

        elif self.prompt_template:
            prompt = self.prompt_template.construct_prompt(messages)

        itr = 0
        while True:
            try:
                if self.idefics_hack:
                    response = self.llm.generate(**inputs, max_new_tokens=self.max_new_tokens)
                    response = self.processor.batch_decode(response, skip_special_tokens=True)[0]
                    response = response.split('Assistant: ')[-1]
                else:
                    response = self.llm(prompt)
                # response = response.split(self.tokenizer.eos_token)[-1]
                return response
            except Exception as e:
                if itr == self.n_retry_server - 1:
                    raise e
                logging.warning(
                    f"Failed to get a response from the server: \n{e}\n"
                    f"Retrying... ({itr+1}/{self.n_retry_server})"
                )
                time.sleep(5)
                itr += 1

    def _llm_type(self):
        return "huggingface"


def _convert_messages_to_dict(messages):
    """
    Converts a list of message objects into a list of dictionaries, categorizing each message by its role.

    Each message is expected to be an instance of one of the following types: SystemMessage, HumanMessage, AIMessage.
    The function maps each message to its corresponding role ('system', 'user', 'assistant') and formats it into a dictionary.

    Args:
        messages (list): A list of message objects.

    Returns:
        list: A list of dictionaries where each dictionary represents a message and contains 'role' and 'content' keys.

    Raises:
        ValueError: If an unsupported message type is encountered.

    Example:
        >>> messages = [SystemMessage("System initializing..."), HumanMessage("Hello!"), AIMessage("How can I assist?")]
        >>> _convert_messages_to_dict(messages)
        [
            {"role": "system", "content": "System initializing..."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "How can I assist?"}
        ]
    """

    # Mapping of message types to roles
    message_type_to_role = {
        SystemMessage: "system",
        HumanMessage: "user",
        AIMessage: "assistant",
    }

    chat = []
    for message in messages:
        message_role = message_type_to_role.get(type(message))
        if message_role:
            chat.append({"role": message_role, "content": message.content})
        else:
            raise ValueError(f"Message type {type(message)} not supported")

    return chat
