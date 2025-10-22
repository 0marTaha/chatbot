import streamlit as st
from F4_dialog_system_Final import load_restaurants, run_dialog_system, train_or_load_classifier, DialogContext, state_transition, formal_templates, informal_templates
import os

# --- Load data ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
restaurant_file = os.path.join(BASE_DIR, "restaurant_info.csv")
restaurants = load_restaurants(restaurant_file)

food_vocab = sorted(set(r["food"] for r in restaurants if r["food"]))
area_vocab = sorted(set(r["area"] for r in restaurants if r["area"]))
price_vocab = sorted(set(r["price"] for r in restaurants if r["price"]))

# --- Streamlit page config ---
st.set_page_config(page_title="Restaurant Chatbot", page_icon="🍽️", layout="centered")

st.title("🍽️ Restaurant Recommendation Chatbot")
st.write("Chat with a simple restaurant recommendation system.")

# Initialize session state
if "context" not in st.session_state:
    st.session_state.context = DialogContext()
    st.session_state.use_formal = False
    st.session_state.allow_ack = False
    st.session_state.ack_prob = 0.7
    st.session_state.clf = train_or_load_classifier(retrain=False)
    st.session_state.templates = informal_templates

# Sidebar settings
st.sidebar.header("⚙️ Settings")
st.session_state.use_formal = st.sidebar.checkbox("Use formal language", value=False)
st.session_state.allow_ack = st.sidebar.checkbox("Allow acknowledgements", value=False)
ack_prob = st.sidebar.slider("Acknowledgement probability", 0.0, 1.0, 0.7)
st.session_state.ack_prob = ack_prob

st.session_state.templates = formal_templates if st.session_state.use_formal else informal_templates

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": st.session_state.templates["welcome"]}]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    sys_resp, st.session_state.context = state_transition(
        st.session_state.context,
        prompt,
        st.session_state.clf,
        restaurants,
        food_vocab,
        area_vocab,
        price_vocab,
        first_pref_suggestion=True,
        ask_confirm_each=False,
        templates=st.session_state.templates,
        allow_restart=True
    )
    st.session_state.messages.append({"role": "assistant", "content": sys_resp})
    st.rerun()
