



import streamlit as st
import requests
import random
from transformers import pipeline
from urllib.parse import quote

# 1. Page Configuration
st.set_page_config(page_title="Emotion Story Engine", page_icon="📖", layout="wide")

# 2. Load and Cache the Emotion Engine
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base", 
        top_k=1
    )

emotion_classifier = load_emotion_model()

# 3. Style Mapping for Anti-Distortion Faces
STYLE_MAP = {
    "joy": "clear happy face, authentic smile, sharp eyes, natural sunlight, candid iPhone photo, 8k resolution, symmetrical features",
    "sadness": "clear face, quiet melancholic expression, soft window light, natural skin texture, unposed documentary photo, sharp focus on eyes",
    "fear": "clear focused face, wide eyes, sharp facial features, low indoor lighting, realistic human anatomy, handheld snapshot",
    "anger": "sharp clear face, intense gaze, realistic skin, indoor lighting, unpolished candid photo, no distortions, symmetrical face",
    "surprise": "clear shocked face, sharp focus, eyebrows raised, authentic human reaction, natural park lighting, realistic anatomy",
    "neutral": "clear normal face, unposed headshot, natural daylight, ordinary person, sharp details, realistic skin tone, clear eyes"
}

# 4. Initialize Session State for History
if "history" not in st.session_state:
    st.session_state.history = []

# 5. Sidebar for Controls & Scenarios
st.sidebar.title("📖 Story Controls")


if st.sidebar.button("🗑️ Reset Story"):
    st.session_state.history = []
    st.session_state.current_prompt = ""
    st.rerun()

# 6. Main Interface
st.title("Emotion-Aware Story Engine")
st.write("Craft realistic daily-life stories with clear, human-like visuals.")

# UI Logic for input
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""

prompt = st.text_area("What happens next?", value=st.session_state.current_prompt, height=100)

if st.button("Generate Realistic Scene"):
    if prompt:
        with st.spinner("Analyzing emotion and drawing scene..."):
            # A. Detect Emotion
            emo_res = emotion_classifier(prompt)[0][0]
            emotion = emo_res['label']

            # B. Generate Story Paragraph
            instruction = (
                f"Continue this story in a clear, meaningful paragraph using simple, "
                f"easy-to-understand words. Make it feel like real daily life. "
                f"The mood is {emotion}. Original thought: {prompt}"
            )
            text_api_url = f"https://text.pollinations.ai/{quote(instruction)}?model=openai"
            story_ext = requests.get(text_api_url).text.strip()

            # C. Generate Image
            tech_style = STYLE_MAP.get(emotion, "clear face, realistic photography")
            image_prompt = (
                f"A high-quality, realistic photo of a normal human being: {prompt}. "
                f"{tech_style}, clear eyes, symmetrical facial structure, natural lighting, "
                f"no distortions, 8k resolution."
            )
            image_url = f"https://image.pollinations.ai/prompt/{quote(image_prompt)}?width=1024&height=768&model=flux&nologo=true"

            # Store in History
            st.session_state.history.append({
                "mood": emotion,
                "input": prompt,
                "story": story_ext,
                "image": image_url
            })
    else:
        st.warning("Please enter a prompt first!")

# 7. Display Results (Newest First)
for item in reversed(st.session_state.history):
    with st.container(border=True):
        st.subheader(f"🎭 Mood: {item['mood'].upper()}")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Prompt:** {item['input']}")
            st.write(item['story'])
        with col2:
            st.image(item['image'], use_container_width=True)


