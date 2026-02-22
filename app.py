import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
def load_model(model_id="Qwen/Qwen3-8B"):
    """
    Caches the model so it only loads once.
    Uses 4-bit quantization to fit into free hosting RAM (e.g., 16GB).
    """
    # Configure 4-bit quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def main():
    st.title("🎓 Tamil-English Language Tutor")
    st.subheader("Learn English through Tamil with AI")

    # Load Model
    with st.spinner("Initializing AI Teacher (4-bit Mode)... This may take a moment."):
        try:
            model, tokenizer = load_model()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": (
                "You are an encouraging English Teacher for Tamil speakers. "
                "Explain English grammar and vocabulary primarily in Tamil. "
                "Provide clear English examples. If the user makes a mistake, gently correct them in Tamil."
            )}
        ]

    # Display Chat History
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a grammar question (e.g., Explain 'Present Tense' in Tamil)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Teacher is typing...")

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
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            response_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            full_response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
            
            response_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
