# Multi-Model LLM Playground

A Streamlit application that allows you to send the same prompt to multiple LLM providers and compare their results side-by-side.

## Features

- **Multi-Provider Support**: Compare responses from OpenAI, Groq, Google Gemini, OpenRouter, NVIDIA, Mistral, and Ollama (local and cloud)
- **Dynamic Model Registry**: Automatically fetch all available models from provider APIs (600+ models)
- **Static Model Registry**: Use a curated list of popular models for quick comparisons
- **Provider Filtering**: Filter models by provider when using dynamic registry
- **Model Search**: Search through available models by name or ID
- **Concurrent Execution**: Run multiple models simultaneously for faster comparison
- **Performance Metrics**: View token usage, latency, and estimated costs for each model
- **Comparison Table**: Side-by-side comparison of all model responses with detailed metrics
- **Flexible Configuration**: Select any combination of registered models
- **Global Model Settings**: Apply a shared system prompt and temperature to every selected model

## Supported Providers

| Provider | API Key Required | Models Available |
|----------|------------------|-----------------|
| OpenAI | `OPENAI_API_KEY` | GPT-4o, GPT-4o-mini, GPT-4, etc. |
| Groq | `GROQ_API_KEY` | GPT-OSS-20B, Mixtral, LLaMA, etc. |
| Google Gemini | `GEMINI_APIKEY` | Gemini 2.5, Gemma, etc. |
| OpenRouter | `OPENROUTER_API_KEY` | 200+ models from various providers |
| NVIDIA | `NVIDIA_API_KEY` | Nemotron, LLaMA, Mixtral, etc. |
| Mistral | `MISTRAL_API_KEY` | Mistral Large, Mixtral, etc. |
| Ollama (Local) | None (requires Ollama installed) | Models installed locally |
| Ollama (Cloud) | `OLLAMA_API_KEY` | Models available on Ollama Cloud |

## Registry Modes

The application supports two registry modes:

### Static Registry (Default)
- Uses a curated list of popular models
- Faster startup time (no API calls needed)
- Recommended for quick comparisons and development
- Configured models:
  - gpt-4o, gpt-4o-mini (OpenAI)
  - groq-gpt-oss-20b (Groq)
  - gemini-3.8-flash (Gemini)
  - ollama-local (Ollama)
  - mistral-medium-latest (Mistral)
  - nvidia/nemotron-3-nano-omni-30b-a3b-reasoning (NVIDIA)

### Dynamic Registry
- Automatically fetches all available models from provider APIs
- Access to 600+ models from all configured providers
- Provider filtering and search capabilities
- Always up-to-date with the latest models
- Requires API keys to be configured
- Slightly slower startup (fetches models on first load)
- Enable by setting `USE_DYNAMIC_REGISTRY=true` in `.env` or toggle in the app

**When to use each mode:**
- Use **Static Registry** for: quick tests, development, when you only need a few specific models
- Use **Dynamic Registry** for: exploring available models, testing many different models, accessing the latest model releases

## Prerequisites

- Python 3.8 or higher
- For Ollama local: [Ollama](https://ollama.com) installed and running on localhost:11434
- API keys for the providers you want to use

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-model-pilot
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
# Registry Mode
USE_DYNAMIC_REGISTRY=false  # Set to true to enable dynamic model fetching

# API Keys (add only the ones you need)
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
OLLAMA_API_KEY=your_ollama_api_key
GEMINI_APIKEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
NVIDIA_API_KEY=your_nvidia_api_key
MISTRAL_API_KEY=your_mistral_api_key

# Ollama Configuration
OLLAMA_LOCAL_HOST=http://localhost:11434
OLLAMA_CLOUD_HOST=https://ollama.com
OLLAMA_CLOUD_MODEL=gemma4:31b
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### How to Use

1. **Choose Registry Mode**: Toggle "Use Dynamic Registry" in the sidebar to switch between static and dynamic modes
2. **Select Models**: In the sidebar, choose which models you want to compare
   - In dynamic mode: Filter by provider and search for specific models
   - In static mode: Select from the curated list
3. **Configure Global Settings**: Expand "Global model settings" to set an optional system prompt and temperature
4. **Enter Prompt**: Type your prompt in the text area
5. **Run Models**: Click the "🚀 Run Models" button
6. **View Results**: See individual model responses and a comparison table with metrics

### Dynamic Registry Features

When using dynamic registry mode, you can:
- **Filter by Provider**: Select specific providers to narrow down the model list
- **Search Models**: Type to search for models by name or ID
- **Refresh Models**: Click "🔄 Refresh Models" to fetch the latest available models from APIs
- **View Provider Info**: See which provider each model comes from in the comparison table

![Global model settings in the sidebar](screenshots/global-settings.png)

## Configuration

### Adding New Models

You have two options for adding new models:

#### Option 1: Use Dynamic Registry (Recommended)
Simply enable dynamic registry mode in the app by toggling "Use Dynamic Registry" in the sidebar. This will automatically fetch all available models from your configured providers.

#### Option 2: Add to Static Registry
To add a new model to the static registry, edit `model_registry.py` and add an entry to the `STATIC_MODEL_REGISTRY` dictionary:

```python
STATIC_MODEL_REGISTRY: dict[str, ModelConfig] = {
    "your-model-key": ModelConfig(
        provider_name="provider",  # openai, groq, gemini, openrouter, mistral, nvidia, ollama_local, ollama_cloud
        model_id="model-id",
        cost_per_input_token=0.0,  # USD per token or None
        cost_per_output_token=0.0,  # USD per token or None
        average_latency_ms=1000,
        quality_tier="high",  # high, medium, or low
    ),
}
```

### Provider Configuration

#### OpenAI
- Set `OPENAI_API_KEY` in `.env`
- Base URL: `https://api.openai.com/v1`

#### Groq
- Set `GROQ_API_KEY` in `.env`
- Base URL: `https://api.groq.com/openai/v1`

#### Google Gemini
- Set `GEMINI_APIKEY` in `.env`
- Uses Google's GenAI SDK

#### OpenRouter
- Set `OPENROUTER_API_KEY` in `.env`
- Base URL: `https://openrouter.ai/api/v1`

#### NVIDIA
- Set `NVIDIA_API_KEY` in `.env`
- Base URL: `https://integrate.api.nvidia.com/v1`

#### Mistral
- Set `MISTRAL_API_KEY` in `.env`
- Base URL: `https://api.mistral.ai/v1`

#### Ollama
- **Local**: Ensure Ollama is running (default: `http://localhost:11434`)
  - Set `OLLAMA_LOCAL_HOST` in `.env` if using a different host
- **Cloud**: Set `OLLAMA_API_KEY` and `OLLAMA_CLOUD_HOST` in `.env`
  - Set `OLLAMA_CLOUD_MODEL` to specify the default model

## Metrics Explained

- **Input Tokens**: Number of tokens in the prompt
- **Output Tokens**: Number of tokens in the response
- **Total Tokens**: Sum of input and output tokens
- **Latency**: Time taken for the model to generate the response (in milliseconds)
- **Cost**: Estimated cost based on token usage (USD)
- **Estimated Usage**: Flag indicating if token counts were estimated (when provider doesn't return usage data)

## Architecture

### Model Registry System

The application uses a flexible model registry system that supports both static and dynamic model discovery:

```
model_registry.py
├── ModelConfig (dataclass)          # Configuration for each model
├── Response (dataclass)             # Response data structure
├── STATIC_MODEL_REGISTRY           # Hardcoded popular models
├── build_dynamic_registry()        # Fetches models from APIs
│   ├── fetch_openai_compatible_models()  # For OpenAI, Groq, OpenRouter, Mistral, NVIDIA
│   ├── fetch_gemini_models()             # For Google Gemini
│   └── fetch_ollama_models()             # For Ollama (local/cloud)
└── send_request()                   # Unified request handler
```

### Dynamic Registry Flow

1. On app startup (or when toggled), check `USE_DYNAMIC_REGISTRY` flag
2. If true, call `build_dynamic_registry()` which:
   - Iterates through all configured providers
   - Calls each provider's model listing API
   - Creates `ModelConfig` entries for each discovered model
   - Returns a dictionary of all available models
3. Models are cached in Streamlit session state to avoid repeated API calls
4. User can filter by provider and search through models
5. "Refresh Models" button clears cache and re-fetches from APIs

### Request Handling

The `send_request()` function provides a unified interface for all providers:
- Accepts a `ModelConfig` object
- Routes to the appropriate provider based on `provider_name`
- Handles provider-specific authentication and API calls
- Normalizes responses into a standard `Response` format
- Calculates costs based on token usage (when pricing is available)

## Testing Dynamic Registry

To test the dynamic model registry without running the full app:

```bash
python test_dynamic_registry.py
```

This will:
- Fetch all available models from your configured providers
- Display them grouped by provider
- Show the total number of models found

## Project Structure

```
llm-model-pilot/
├── app.py                      # Main Streamlit application
├── model_registry.py           # Model configurations and request handling
├── test_dynamic_registry.py    # Script to test dynamic model fetching
├── .env.example                # Example environment variables
├── .env                        # Your actual environment variables (not in git)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Troubleshooting

### Missing API Key Error
If you see "Missing environment variable: X", ensure you have added the required API key to your `.env` file and reload the app.

### Dynamic Registry Shows No Models
- Verify your API keys are correct in `.env`
- Check that your API keys have the necessary permissions
- Some providers may rate limit model listing requests
- Try refreshing the models using the "🔄 Refresh Models" button

### Ollama Connection Failed
- Ensure Ollama is installed and running: `ollama serve`
- Check that `OLLAMA_LOCAL_HOST` is correct (default: `http://localhost:11434`)
- Verify you can access Ollama: `curl http://localhost:11434/api/tags`

### Model Request Failed
- Check that the model ID is correct for the provider
- Verify your API key has access to the specific model
- Some models may be in beta or have restricted access
- Check your API quota/credits

## License

This project is provided as-is for educational and comparison purposes.
