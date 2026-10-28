
import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from model_registry import (
    MODEL_REGISTRY,
    send_request,
    USE_DYNAMIC_REGISTRY,
    build_dynamic_registry,
    STATIC_MODEL_REGISTRY,
)

st.set_page_config(
    page_title="Multi-Model LLM Playground",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Model LLM Playground")
st.write("Send the same prompt to multiple LLMs and compare their results.")

# Sidebar: registry mode selection
st.sidebar.header("Registry Mode")

use_dynamic = st.sidebar.toggle(
    "Use Dynamic Registry",
    value=USE_DYNAMIC_REGISTRY,
    help="Dynamic registry fetches all available models from provider APIs. Static registry uses a curated list.",
)

# Update registry based on selection
if use_dynamic:
    if "dynamic_registry" not in st.session_state:
        with st.sidebar.spinner("Fetching models from provider APIs..."):
            st.session_state.dynamic_registry = build_dynamic_registry()
    MODEL_REGISTRY = st.session_state.dynamic_registry
    st.info("🔄 Using dynamic model registry (fetched from provider APIs)")
else:
    MODEL_REGISTRY = STATIC_MODEL_REGISTRY
    st.info("📋 Using static model registry")

# Sidebar: model selection
st.sidebar.header("Model Selection")

model_names = list(MODEL_REGISTRY.keys())

# For dynamic registry, add provider filtering
if use_dynamic:
    # Get unique providers
    providers = sorted(set(config.provider_name for config in MODEL_REGISTRY.values()))
    selected_providers = st.sidebar.multiselect(
        "Filter by provider",
        options=providers,
        default=providers[:3],  # Default to first 3 providers
        help="Select providers to filter available models",
    )

    # Filter models by selected providers
    if selected_providers:
        model_names = [
            name for name, config in MODEL_REGISTRY.items()
            if config.provider_name in selected_providers
        ]
    else:
        model_names = []

    st.sidebar.caption(f"Showing {len(model_names)} models from {len(selected_providers)} providers")

    # Add search/filter for models
    search_query = st.sidebar.text_input(
        "🔍 Search models",
        placeholder="Type to filter models...",
        help="Search by model name or ID",
    )

    if search_query:
        search_lower = search_query.lower()
        model_names = [
            name for name in model_names
            if search_lower in name.lower() or search_lower in MODEL_REGISTRY[name].model_id.lower()
        ]
        st.sidebar.caption(f"Filtered to {len(model_names)} models")

    # Add refresh button
    if st.sidebar.button("🔄 Refresh Models"):
        with st.sidebar.spinner("Refreshing models from provider APIs..."):
            st.session_state.dynamic_registry = build_dynamic_registry()
            st.rerun()

# Set default models based on what's available
if use_dynamic:
    # For dynamic registry, select up to 5 models from different providers
    default_models = []
    providers_seen = set()
    for name in model_names:
        config = MODEL_REGISTRY[name]
        if config.provider_name not in providers_seen:
            default_models.append(name)
            providers_seen.add(config.provider_name)
        if len(default_models) >= 5:
            break
else:
    # For static registry, use the original defaults
    default_models = [
        name for name in [
            "gpt-4o-mini",
            "groq-gpt-oss-20b",
            "gemini-3.8-flash",
            "ollama-local",
            "mistral-medium-latest",
        ]
        if name in MODEL_REGISTRY
    ]

selected_models = st.sidebar.multiselect(
    "Select models to run",
    options=model_names,
    default=default_models,
)

with st.sidebar.expander("Global model settings", expanded=False):
    system_prompt = st.text_area(
        "System prompt",
        placeholder="Optional instructions applied to every model...",
        help="This instruction is sent before your prompt to each selected model.",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Lower values make output more deterministic; higher values increase variation.",
    )

# Prompt input
prompt = st.text_area(
    "Enter your prompt",
    placeholder="Explain RAG and Qdrant in simple terms...",
    height=150,
)

run_button = st.button(
    "🚀 Run Models",
    type="primary",
    disabled=not prompt.strip() or not selected_models,
)

if run_button:
    results = {}

    progress = st.progress(0)
    status = st.empty()

    def run_one(name):
        config = MODEL_REGISTRY[name]
        return send_request(
            prompt,
            config,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    with ThreadPoolExecutor(
        max_workers=min(len(selected_models), 5)
    ) as executor:
        futures = {
            executor.submit(run_one, name): name
            for name in selected_models
        }

        for index, future in enumerate(as_completed(futures), start=1):
            name = futures[future]

            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = exc

            progress.progress(index / len(selected_models))
            status.write(
                f"Completed {index}/{len(selected_models)} models"
            )

    progress.empty()
    status.empty()

    # Summary metrics
    successful = {
        name: result
        for name, result in results.items()
        if not isinstance(result, Exception)
    }

    if successful:
        total_tokens = sum(
            result.tokens_used for result in successful.values()
        )
        total_latency = sum(
            result.latency_ms for result in successful.values()
        )
        known_costs = [
            result.cost
            for result in successful.values()
            if result.cost is not None
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric("Successful models", len(successful))
        col2.metric("Total tokens", f"{total_tokens:,}")
        col3.metric(
            "Known total cost",
            f"${sum(known_costs):.8f}",
        )

        st.caption(
            "Total latency below is measured per model. "
            "Requests run concurrently, so wall-clock time differs."
        )

    # Individual model responses
    st.subheader("Model Responses")

    for name in selected_models:
        result = results.get(name)

        with st.expander(f"📦 {name}", expanded=True):
            if isinstance(result, Exception):
                st.error(f"Request failed: {result}")
                continue

            st.markdown(result.output_text)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Input tokens", result.input_tokens)
            c2.metric("Output tokens", result.output_tokens)
            c3.metric("Latency", f"{result.latency_ms:.0f} ms")
            c4.metric(
                "Cost",
                f"${result.cost:.8f}"
                if result.cost is not None
                else "Unknown",
            )

            if result.usage_estimated:
                st.warning(
                    "Token counts include estimates because "
                    "provider usage was unavailable."
                )

            st.caption(f"Model ID: {result.model_id}")
            if use_dynamic:
                st.caption(f"Provider: {MODEL_REGISTRY[name].provider_name}")

    # Comparison table
    if successful:
        st.subheader("Comparison Table")

        comparison = []
        for name, result in successful.items():
            row = {
                "Registry key": name,
                "Model ID": result.model_id,
                "Input tokens": result.input_tokens,
                "Output tokens": result.output_tokens,
                "Total tokens": result.tokens_used,
                "Latency (ms)": round(result.latency_ms, 2),
                "Cost (USD)": result.cost,
                "Estimated usage": result.usage_estimated,
            }
            # Add provider info for dynamic mode
            if use_dynamic:
                row["Provider"] = MODEL_REGISTRY[name].provider_name
            comparison.append(row)

        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
        )
