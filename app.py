import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Page configuration
st.set_page_config(page_title="Tamil-English AI Tutor", page_icon="🎓")

# Custom CSS for a better look
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stChatInputContainer { padding-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model(model_id="Qwen/Qwen3-8B-Instruct"):
    """
    Caches the model so it only loads once when the app starts.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Using 4-bit quantization (bitsandbytes) is recommended for free hosting tiers
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def main():
    st.title("🎓 Tamil-English Language Tutor")
    st.subheader("Learn English through Tamil with AI")

    # Load Model
    with st.spinner("Initializing AI Teacher... This may take a moment."):
        model, tokenizer = load_model()

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": (
                "You are an encouraging English Teacher for Tamil speakers. "
                "Explain concepts in Tamil. Provide English examples. "
                "If the user makes a mistake in English, gently correct them in Tamil."
            )}
        ]

    # Display Chat History
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a grammar question (e.g., Explain 'Past Tense' in Tamil)"):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Teacher is typing...")

            # Format prompt for the model
            input_text = tokenizer.apply_chat_template(
                st.session_state.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode response
            response_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            full_response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
            
            response_placeholder.markdown(full_response)
            
        # Add assistant response to state
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
