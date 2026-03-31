import os
from typing import Optional
import tiktoken
from openai import AsyncOpenAI

from .model import ModelProvider


class LocalModel(ModelProvider):
    """
    A wrapper class for interacting with local models via OpenAI-compatible API (like vLLM, Ollama, etc.),
    providing methods to encode text, generate prompts, evaluate models.

    Attributes:
        model_name (str): The name of the model to use.
        base_url (str): The base URL for the local model API endpoint.
        model (AsyncOpenAI): An instance of the AsyncOpenAI client configured for local endpoint.
        tokenizer: A tokenizer instance for encoding and decoding text to and from token representations.
    """
        
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens=300, temperature=0)

    def __init__(self,
                 model_name: str = "local-model",
                 base_url: str = None,
                 model_kwargs: dict = None):
        """
        Initializes the Local Model provider with a specific model and endpoint.

        Args:
            model_name (str): The name of the local model to use.
            base_url (str): The base URL of the local API endpoint (e.g., "http://localhost:8000/v1")
            model_kwargs (dict): Model configuration. Defaults to {max_tokens: 300, temperature: 0}.
        
        Raises:
            ValueError: If base_url is not provided and NIAH_MODEL_BASE_URL is not found in the environment.
        """
        # Get base URL from parameter or environment
        self.base_url = base_url or os.getenv('NIAH_MODEL_BASE_URL')
        if not self.base_url:
            raise ValueError("base_url must be provided or NIAH_MODEL_BASE_URL must be in env. "
                           "Example: http://localhost:8000/v1")

        # API key is optional for local models but required by the client
        api_key = os.getenv('NIAH_MODEL_API_KEY', 'dummy-key')
        
        self.model_name = model_name
        self.model_kwargs = model_kwargs or self.DEFAULT_MODEL_KWARGS
        self.api_key = api_key
        
        # Initialize client with local endpoint
        self.model = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # Try to use tiktoken, fallback to gpt-3.5-turbo encoding if model-specific not available
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # For custom models, use cl100k_base (GPT-3.5/4 tokenizer) as approximation
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    async def evaluate_model(self, prompt: str) -> str:
        """
        Evaluates a given prompt using the local model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt.
        """
        response = await self.model.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            **self.model_kwargs
        )
        return response.choices[0].message.content
    
    def generate_prompt(self, context: str, retrieval_question: str) -> list[dict[str, str]]:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the structured prompt.
        """
        return [{
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "user",
                "content": context
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings"
            }]
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens[:context_length])
