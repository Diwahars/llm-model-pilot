
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from model_registry import MODEL_REGISTRY, send_request

st.set_page_config(
    page_title="Multi-Model LLM Playground",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Model LLM Playground")
st.write("Send the same prompt to multiple LLMs and compare their results.")

# Sidebar: model selection
st.sidebar.header("Model Selection")

model_names = list(MODEL_REGISTRY.keys())

selected_models = st.sidebar.multiselect(
    "Select models to run",
    options=model_names,
    default=[
        name for name in [
            "gpt-4o-mini",
            "groq-gpt-oss-20b",
            "gemini-3.8-flash",
            "ollama-local",
        ]
        if name in MODEL_REGISTRY
    ],
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
        return send_request(prompt, config)

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

    # Comparison table
    if successful:
        st.subheader("Comparison Table")

        comparison = []
        for name, result in successful.items():
            comparison.append({
                "Registry key": name,
                "Model ID": result.model_id,
                "Input tokens": result.input_tokens,
                "Output tokens": result.output_tokens,
                "Total tokens": result.tokens_used,
                "Latency (ms)": round(result.latency_ms, 2),
                "Cost (USD)": result.cost,
                "Estimated usage": result.usage_estimated,
            })

        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,̥
        )
