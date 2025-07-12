import streamlit as st
import asyncio
import os
from together import AsyncTogether, Together

# Define Models and Aggregator System Prompt
reference_models = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "Open-Orca/Mistral-7B-OpenOrca",
]
aggregator_model = "mistralai/Mistral-7B-Instruct-v0.1"
aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given responses but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, and meets the highest standards of accuracy and reliability."""

# Streamlit App Setup and API Key Input
st.title("Mixture-of-Agents LLM App")
together_api_key = st.text_input("Enter your Together API Key:", type="password")

# Initialize Together AI Clients
if together_api_key:
    os.environ["TOGETHER_API_KEY"] = together_api_key
    client = Together(api_key=together_api_key)
    async_client = AsyncTogether(api_key=together_api_key)

# Asynchronous Function to Run an LLM
async def run_llm(model, temperature=0.7):
    try:
        response = await async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=512,
        )
        return f"{model} (temp={temperature})", response.choices[0].message.content
    except Exception as e:
        return f"{model} (temp={temperature})", f"Error with {model}: {str(e)}"

# Main Asynchronous Function to Run All Models and Aggregate
async def main():
    try:
        # Run the same model with different temperatures for variety
        model_calls = [
            run_llm("mistralai/Mistral-7B-Instruct-v0.1", 0.3),  # Conservative
            run_llm("mistralai/Mistral-7B-Instruct-v0.1", 0.7),  # Balanced
            run_llm("mistralai/Mistral-7B-Instruct-v0.1", 0.9),  # Creative
        ]
        
        # Run all model calls asynchronously
        results = await asyncio.gather(*model_calls)
        
        # Display individual model responses
        st.subheader("Individual Model Responses:")
        successful_responses = []
        
        for model, response in results:
            with st.expander(f"Response from {model}"):
                if response.startswith("Error with"):
                    st.error(response)
                else:
                    st.write(response)
                    successful_responses.append(response)
        
        # Only aggregate if we have successful responses
        if successful_responses:
            st.subheader("Aggregated Response:")
            try:
                finalStream = client.chat.completions.create(
                    model=aggregator_model,
                    messages=[
                        {"role": "system", "content": aggregator_system_prompt},
                        {"role": "user", "content": " ".join(successful_responses)},
                    ],
                    stream=True,
                )
                # Display aggregated response in a streaming fashion
                response_container = st.empty()
                full_response = ""
                for chunk in finalStream:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    response_container.markdown(full_response + "▌")
                response_container.markdown(full_response)
            except Exception as e:
                st.error(f"Error with aggregator model: {str(e)}")
        else:
            st.error("No successful responses from reference models to aggregate.")
            
    except Exception as e:
        st.error(f"General error: {str(e)}")

# User Interface and Main Function Trigger
user_prompt = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_prompt and together_api_key:
        with st.spinner("Generating responses from multiple models..."):
            asyncio.run(main())
    elif not together_api_key:
        st.error("Please enter your Together API Key first.")
    else:
        st.warning("Please enter a question.")
