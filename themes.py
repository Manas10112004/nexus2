import streamlit as st

# --- THEME DEFINITIONS ---
THEMES = {
    "🌿 Eywa (Avatar)": {
        "primary": "#00e5ff",
        "secondary": "#7c4dff",
        "bg_color": "#050a0e",
        "bg_gradient": "linear-gradient(to top, #0f2027, #203a43, #050a0e)",
        "sidebar_bg": "#020406",
        "font_header": "sans-serif",
        "font_body": "sans-serif",
        "user_avatar": "🧑‍🚀",
        "ai_avatar": "🧬"
    },
    "🚕 Bumblebee": {
        "primary": "#FFD700",
        "secondary": "#000000",
        "bg_color": "#050505",
        "bg_gradient": "radial-gradient(circle at 50% 50%, #1a1a1a 1px, transparent 1px)",
        "sidebar_bg": "#111111",
        "font_header": "monospace",
        "font_body": "monospace",
        "user_avatar": "👤",
        "ai_avatar": "🚕"
    },
    "🔵 Jarvis": {
        "primary": "#00a8ff",
        "secondary": "#005073",
        "bg_color": "#000000",
        "bg_gradient": "radial-gradient(circle at center, #001f3f 0%, #000000 100%)",
        "sidebar_bg": "#000810",
        "font_header": "sans-serif",
        "font_body": "sans-serif",
        "user_avatar": "🤵",
        "ai_avatar": "💿"
    },
    "🏴‍☠️ One Piece": {
        "primary": "#FF4500",
        "secondary": "#FFD700",
        "bg_color": "#001a33",
        "bg_gradient": "linear-gradient(135deg, #001a33 0%, #004080 100%)",
        "sidebar_bg": "#000d1a",
        "font_header": "serif",
        "font_body": "sans-serif",
        "user_avatar": "🍖",
        "ai_avatar": "🏴‍☠️"
    },
    "🥋 Dragon Ball": {
        "primary": "#FF8C00",
        "secondary": "#0057B7",
        "bg_color": "#1a0b00",
        "bg_gradient": "linear-gradient(to bottom right, #ffbb00, #ff8c00, #1a0b00)",
        "sidebar_bg": "#1a0b00",
        "font_header": "sans-serif",
        "font_body": "sans-serif",
        "user_avatar": "🥡",
        "ai_avatar": "🐉"
    },
    "☣️ Overflow": {
        "primary": "#ff00ff",
        "secondary": "#00ff00",
        "bg_color": "#000000",
        "bg_gradient": "repeating-linear-gradient(45deg, #000 0, #000 10px, #111 10px, #111 20px)",
        "sidebar_bg": "#050505",
        "font_header": "monospace",
        "font_body": "monospace",
        "user_avatar": "💀",
        "ai_avatar": "👁️"
    }
}


def inject_theme_css(theme_name):
    # Fallback safety
    if theme_name not in THEMES:
        theme_name = "🌿 Eywa (Avatar)"
    t = THEMES[theme_name]

    # We construct the CSS string here to avoid syntax errors
    css_code = f"""
    <style>
        .stApp {{
            background-color: {t['bg_color']};
            background-image: {t['bg_gradient']};
            color: #e0e0e0;
        }}
        [data-testid="stSidebar"] {{
            background-color: {t['sidebar_bg']};
            border-right: 1px solid {t['primary']};
        }}
        h1, h2, h3 {{
            color: {t['primary']} !important;
            text-shadow: 0px 0px 10px {t['primary']}80;
            font-family: {t['font_header']} !important;
        }}
        .stButton > button {{
            border: 1px solid {t['primary']};
            color: {t['primary']};
            background: transparent;
        }}
        .stButton > button:hover {{
            background: {t['primary']};
            color: {t['bg_color']};
        }}
        .stTextInput > div {{
            border-radius: 5px;
            border: 1px solid {t['secondary']};
        }}
    </style>
    """

    st.markdown(css_code, unsafe_allow_html=True)