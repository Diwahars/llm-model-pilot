
import os
import time
from dataclasses import dataclass
from typing import Literal, Optional

import ollama
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()

QualityTier = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class ModelConfig:
    provider_name: str
    model_id: str
    cost_per_input_token: Optional[float]
    cost_per_output_token: Optional[float]
    average_latency_ms: float
    quality_tier: QualityTier


@dataclass
class Response:
    output_text: str
    input_tokens: int
    output_tokens: int
    tokens_used: int
    latency_ms: float
    cost: Optional[float]
    model_id: str
    usage_estimated: bool = False


# Prices are USD per token.
# Latency values are initial estimates, not measured benchmarks.
# None means pricing is not configured.
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        provider_name="openai",
        model_id="gpt-4o",
        cost_per_input_token=2.50 / 1_000_000,
        cost_per_output_token=10.00 / 1_000_000,
        average_latency_ms=1200,
        quality_tier="high",
    ),
    "gpt-4o-mini": ModelConfig(
        provider_name="openai",
        model_id="gpt-4o-mini",
        cost_per_input_token=0.15 / 1_000_000,
        cost_per_output_token=0.60 / 1_000_000,
        average_latency_ms=600,
        quality_tier="medium",
    ),
    "groq-gpt-oss-20b": ModelConfig(
        provider_name="groq",
        model_id="openai/gpt-oss-20b",
        cost_per_input_token=0.075 / 1_000_000,
        cost_per_output_token=0.30 / 1_000_000,
        average_latency_ms=600,
        quality_tier="high",
    ),
    "gemini-3.8-flash": ModelConfig(
        provider_name="gemini",
        model_id="gemini-3.8-flash",
        cost_per_input_token=0.30 / 1_000_000,
        cost_per_output_token=2.50 / 1_000_000,
        average_latency_ms=700,
        quality_tier="high",
    ),
    "openrouter": ModelConfig(
        provider_name="openrouter",
        model_id="nvidia/nemotron-3-ultra-550b-a55b:free",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        average_latency_ms=700,
        quality_tier="medium",
    ),
    "ollama-local": ModelConfig(
        provider_name="ollama_local",
        model_id="llama3.2:latest",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        average_latency_ms=800,
        quality_tier="medium",
    ),
    "ollama-cloud": ModelConfig(
        provider_name="ollama_cloud",
        model_id=os.getenv(
            "OLLAMA_CLOUD_MODEL", "gemma4:31b"
        ),
        cost_per_input_token=None,
        cost_per_output_token=None,
        average_latency_ms=1200,
        quality_tier="high",
    ),
}


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Missing environment variable: {name}"
        )
    return value


def send_request(prompt: str, model_config: ModelConfig) -> Response:
    """Send a prompt to any registered provider."""

    start = time.perf_counter()
    provider = model_config.provider_name
    model = model_config.model_id

    output_text = ""
    input_tokens = None
    output_tokens = None

    if provider in ("openai", "groq","openrouter"):
        if provider == "openai":
            client = OpenAI(
                api_key=required_env("OPENAI_API_KEY")
            )
        elif provider == "openrouter":
            client = OpenAI(
                api_key=required_env("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            client = OpenAI(
                api_key=required_env("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )

        result = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        output_text = result.choices[0].message.content or ""

        if result.usage is not None:
            input_tokens = result.usage.prompt_tokens
            output_tokens = result.usage.completion_tokens

    elif provider == "gemini":
        client = genai.Client(
            api_key=required_env("GEMINI_APIKEY")
        )

        result = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
            ),
        )

        output_text = result.text or ""

        usage = result.usage_metadata
        if usage is not None:
            input_tokens = usage.prompt_token_count
            output_tokens = usage.candidates_token_count

    elif provider in ("ollama_local", "ollama_cloud"):
        if provider == "ollama_local":
            client = ollama.Client(
                host=os.getenv(
                    "OLLAMA_LOCAL_HOST",
                    "http://localhost:11434",
                )
            )
        else:
            client = ollama.Client(
                host=os.getenv(
                    "OLLAMA_CLOUD_HOST",
                    "https://ollama.com",
                ),
                headers={
                    "Authorization": (
                        "Bearer "
                        + required_env("OLLAMA_API_KEY")
                    )
                },
            )

        result = client.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        output_text = result.message.content or ""
        input_tokens = result.prompt_eval_count
        output_tokens = result.eval_count

    else:
        raise ValueError(
            f"Unsupported provider: {provider}"
        )

    # Estimate missing token counts only.
    # Character-based estimates are approximate.
    usage_estimated = (
        input_tokens is None or output_tokens is None
    )

    if input_tokens is None:
        input_tokens = max(1, len(prompt) // 4)

    if output_tokens is None:
        output_tokens = (
            max(1, len(output_text) // 4)
            if output_text
            else 0
        )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    # Calculate cost only when both rates are configured.
    if (
        model_config.cost_per_input_token is not None
        and model_config.cost_per_output_token is not None
    ):
        cost = (
            input_tokens
            * model_config.cost_per_input_token
            + output_tokens
            * model_config.cost_per_output_token
        )
    else:
        cost = None

    return Response(
        output_text=output_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokens_used=input_tokens + output_tokens,
        latency_ms=round(latency_ms, 2),
        cost=round(cost, 10) if cost is not None else None,
        model_id=model,
        usage_estimated=usage_estimated,
    )