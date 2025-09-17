# 
import os
from typing import Tuple

import openai
import backoff
import warnings


def get_client(model_name: str) -> openai.OpenAI:
    """Initialize and return the API client based on the model type"""

    if model_name.startswith("gpt-"):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://api.openai.com/v1"
    elif model_name.startswith(("o1-", "o3-", "o4-")):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://api.openai.com/v1"
    elif model_name.startswith("deepseek-"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = "https://api.deepseek.com"
    elif model_name.startswith("ollama/"):
        # https://ollama.com/blog/openai-compatibility
        base_url = 'http://localhost:11434/v1'
        api_key='ollama', # required, but unused
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    return client


# NOTE: price per 1M token, recorded on 2025-03-23
API_PRICING = {
    # OpenAI models
    "gpt-4o-2024-05-13": {"input": 5.0, "output": 15.0},
    "gpt-4o-2024-08-06": {"input": 2.5, "output": 10.0},
    "gpt-4o-2024-11-20": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.6},
    "o1-2024-12-17": {"input": 15.0, "output": 60.0},
    "o1-pro-2025-03-19": {"input": 150.0, "output": 600.0},
    "o1-mini-2024-09-12": {"input": 1.10, "output": 4.40},
    "o3-mini-2025-01-31": {"input": 1.10, "output": 4.40},
    "o4-mini-2025-04-16": {"input": 1.10, "output": 4.40},
    # Deepseek models  (using standard price. discount price would be lower)
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}


def calculate_api_cost(prompt_tokens, completion_tokens, model) -> float:
    if model in API_PRICING:
        input_cost = (prompt_tokens / 1_000_000) * API_PRICING[model]["input"]
        output_cost = (completion_tokens / 1_000_000) * API_PRICING[model]["output"]
        return input_cost + output_cost
    else:
        warnings.warn(f"Unknown model: {model}. No pricing information.", UserWarning)
        return float('nan')
    

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def query_llm(client: openai.OpenAI, model: str, 
              system_message: str, user_message: str, 
              chat_kwargs: dict = {}) -> Tuple[openai.ChatCompletion, float]:
    
    if model.startswith("ollama/"):
        model_name = model.split("ollama/")[1]  # local ollama models
    else:
        model_name = model

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        **chat_kwargs,
    )

    if model.startswith("ollama/"):
        cost = 0.0  # locally hosted model
    else:
        cost = calculate_api_cost(
            prompt_tokens=response.usage.prompt_tokens, 
            completion_tokens=response.usage.completion_tokens, 
            model=model_name
        )

    return response, cost