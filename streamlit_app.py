import streamlit as st
import os
from F4_dialog_system_Final import (
    load_restaurants,
    train_or_load_classifier,
    DialogContext,
    state_transition,
    formal_templates,
    informal_templates,
)

# ---------------------------
# 📂 Load restaurant data
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
restaurant_file = os.path.join(BASE_DIR, "restaurant_info.csv")

if not os.path.exists(restaurant_file):
    st.error(f"❌ Restaurant file not found at: {restaurant_file}")
    st.stop()

restaurants = load_restaurants(restaurant_file)
food_vocab = sorted(set(r["food"] for r in restaurants if r["food"]))
area_vocab = sorted(set(r["area"] for r in restaurants if r["area"]))
price_vocab = sorted(set(r["price"] for r in restaurants if r["price"]))

# ---------------------------
# 🌐 Page setup
# ---------------------------
st.set_page_config(page_title="Restaurant Chatbot", page_icon="🍽️", layout="centered")
st.title("🍽️ Restaurant Recommendation Chatbot")
st.write("Chat with an intelligent restaurant recommendation system.")

# ---------------------------
# 🧠 Initialize session state
# ---------------------------
if "context" not in st.session_state:
    st.session_state.context = DialogContext()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Default config values
defaults = {
    "use_formal": True,
    "allow_ack": True,
    "ack_prob": 0.7,
    "first_pref_suggestion": True,
    "ask_confirm_each": True,
    "allow_restart": True,
    "retrain_classifier": True,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------------------
# ⚙️ Sidebar: Settings
# ---------------------------
st.sidebar.header("⚙️ Chatbot Settings")

st.session_state.use_formal = st.sidebar.radio(
    "Language style",
    options=["Informal", "Formal"],
    index=0,
) == "Formal"

st.session_state.allow_ack = st.sidebar.checkbox(
    "Enable acknowledgements", value=st.session_state.allow_ack
)

st.session_state.ack_prob = st.sidebar.slider(
    "Acknowledgement probability",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.ack_prob,
    step=0.05,
)

st.session_state.first_pref_suggestion = st.sidebar.checkbox(
    "Offer suggestion after first preference",
    value=st.session_state.first_pref_suggestion
)

st.session_state.ask_confirm_each = st.sidebar.checkbox(
    "Ask confirmation for each preference",
    value=st.session_state.ask_confirm_each
)

st.session_state.allow_restart = st.sidebar.checkbox(
    "Allow dialog restart",
    value=st.session_state.allow_restart
)

st.session_state.retrain_classifier = st.sidebar.checkbox(
    "Retrain dialog classifier at start",
    value=st.session_state.retrain_classifier
)

if st.sidebar.button("🔄 Reset conversation"):
    st.session_state.context = DialogContext()
    st.session_state.messages = [
        {"role": "assistant", "content": "🔄 Conversation reset. How can I help you?"}
    ]
    st.rerun()

# ---------------------------
# 🧠 Load classifier and templates
# ---------------------------
templates_choice = formal_templates if st.session_state.use_formal else informal_templates
clf = train_or_load_classifier(retrain=st.session_state.retrain_classifier)

# ---------------------------
# 💬 Initial bot message
# ---------------------------
if len(st.session_state.messages) == 0:
    st.session_state.messages.append({"role": "assistant", "content": templates_choice["welcome"]})

# ---------------------------
# 🪄 Chat display
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# 📨 User input handling
# ---------------------------
# 📨 User input handling
if prompt := st.chat_input("Type your message..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    sys_resp, st.session_state.context = state_transition(
        st.session_state.context,
        prompt,
        clf,
        restaurants,
        food_vocab,
        area_vocab,
        price_vocab,
        first_pref_suggestion=st.session_state.first_pref_suggestion,
        ask_confirm_each=st.session_state.ask_confirm_each,
        templates=templates_choice,
        allow_restart=st.session_state.allow_restart,
    )

    # Display bot message
    st.session_state.messages.append({"role": "assistant", "content": sys_resp})

    # If conversation ended — show feedback form
    if st.session_state.context.state == "goodbye":
        st.session_state.show_feedback_form = True

    st.rerun()

# 📝 After chat: Feedback form
if st.session_state.get("show_feedback_form", False):
    st.divider()
    st.success("✅ Thank you for using the chatbot!")
    st.write("We’d love to hear your feedback. Please fill in this short form:")
    st.markdown(
        "[👉 Fill in the Google Form](https://docs.google.com/forms/d/e/1FAIpQLSeCidHU3piXdW4jcwtCbrZCtOdZ5MI20nB0RWAzdPfCKVqy6Q/viewform)",
        unsafe_allow_html=True
    )

