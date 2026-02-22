import streamlit as st
import google.generativeai as genai

# Page configuration
st.set_page_config(page_title="Tamil-English AI Tutor", page_icon="🎓")

# Custom CSS
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stChatInputContainer { padding-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🎓 Tamil-English Language Tutor")
    st.subheader("Learn English through Tamil with AI")

    # API Key Configuration
    # In Streamlit Cloud, you can add this to "Settings > Secrets" as GEMINI_API_KEY = "your_key"
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    
    if not api_key:
        st.info("Please enter your Gemini API Key in the sidebar to start learning! You can get a free key at https://aistudio.google.com/", icon="🔑")
        return

    # Initialize Gemini
    genai.configure(api_key=api_key)
    
    # Updated to Gemini 2.5 Flash for the latest stable support
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=(
            "You are an encouraging English Teacher for Tamil speakers. "
            "Explain English grammar and vocabulary primarily in Tamil. "
            "Provide clear English examples. If the user makes a mistake in English, "
            "gently correct them using Tamil explanations."
        )
    )

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Start the chat session
        st.session_state.chat = model.start_chat(history=[])

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a grammar question (e.g., How to use 'Have' and 'Has'?)"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Teacher is typing...")
            
            try:
                # Send message to Gemini
                response = st.session_state.chat.send_message(prompt)
                full_response = response.text
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                # Detailed error messaging for debugging
                if "404" in str(e):
                    st.error("Error: The model version was not found. Please ensure you are using the latest 'google-generativeai' package.")
                else:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()