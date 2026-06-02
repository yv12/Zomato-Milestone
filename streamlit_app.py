"""
TasteFinder AI — Streamlit Entry Point
=======================================
Streamlit-based UI for the Zomato Restaurant Discovery Engine.
Replaces the React + FastAPI frontend while reusing the existing
service layer (orchestrator, filter, vector search, LLM client).
"""

import os
import sys
import traceback

# ---------------------------------------------------------------------------
# 1. PATH SETUP — must happen before any app imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# ---------------------------------------------------------------------------
# 2. BRIDGE STREAMLIT SECRETS → os.environ (before config.py is imported)
# ---------------------------------------------------------------------------
import streamlit as st

_SECRET_KEYS = [
    "GROQ_API_KEY", "LLM_MODEL", "EMBEDDING_MODEL_NAME",
    "DATA_PATH", "MAX_CANDIDATES", "BUDGET_LOW_MAX", "BUDGET_MEDIUM_MAX",
]
for _key in _SECRET_KEYS:
    try:
        _val = st.secrets.get(_key)
        if _val is not None:
            os.environ[_key] = str(_val)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 3. APP IMPORTS (after env is configured)
# ---------------------------------------------------------------------------
from app.services.registry import ServiceRegistry
from app.models.domain import UserPreferences

# ---------------------------------------------------------------------------
# 4. PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TasteFinder AI - Zomato",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# 5. CUSTOM CSS — dark theme matching the React frontend
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---------- Import Google Fonts ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;600;700&display=swap');

    /* ---------- Root Variables ---------- */
    :root {
        --bg: #0e1111;
        --surface: #1a1d1d;
        --border: #2a2a30;
        --text: #f0ece8;
        --text-muted: #b5b5be;
        --primary: #8B0000;
        --gold: #d4a825;
    }

    /* ---------- Global Overrides ---------- */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
    }
    [data-testid="stHeader"] {
        background-color: var(--bg) !important;
    }
    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
    }

    /* ---------- Hide default Streamlit chrome ---------- */
    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
    }

    /* ---------- Headings ---------- */
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }

    /* ---------- Card component ---------- */
    .rec-card {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0;
        overflow: hidden;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        height: 480px; /* Force cards to have identical height */
        display: flex;
        flex-direction: column;
    }
    .rec-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(139, 0, 0, 0.2), 0 0 12px rgba(212, 168, 37, 0.1);
    }
    .rec-card-img {
        width: 100%;
        height: 140px;
        object-fit: cover;
        display: block;
    }
    .rec-card-body {
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        flex-grow: 1;
    }
    .rec-card-title {
        font-size: 16px;
        font-weight: 700;
        color: var(--text);
        margin: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .rec-card-meta {
        display: flex;
        flex-wrap: nowrap; /* Prevent chips from wrapping to two rows */
        gap: 6px;
        overflow: hidden;
    }
    .rec-card-chip {
        font-size: 11px;
        color: var(--text-muted);
        background: var(--bg);
        border: 1px solid var(--border);
        padding: 3px 8px;
        border-radius: 6px;
        white-space: nowrap; /* Keep chip contents on a single line */
    }
    .rec-card-explanation {
        font-size: 12px;
        font-weight: 500;
        color: var(--gold);
        background: rgba(212, 168, 37, 0.06);
        border: 1px solid rgba(212, 168, 37, 0.12);
        padding: 8px 12px;
        border-radius: 10px;
        line-height: 1.45;
        height: 110px; /* Fixed height for the AI explanation text block */
        overflow-y: auto; /* Sleek scrolling if text exceeds height */
    }
    /* Elegant custom scrollbar for card explanation block */
    .rec-card-explanation::-webkit-scrollbar {
        width: 4px;
    }
    .rec-card-explanation::-webkit-scrollbar-track {
        background: transparent;
    }
    .rec-card-explanation::-webkit-scrollbar-thumb {
        background: rgba(212, 168, 37, 0.15);
        border-radius: 4px;
    }
    .rec-card-explanation::-webkit-scrollbar-thumb:hover {
        background: rgba(212, 168, 37, 0.3);
    }
    .rec-card-explanation::before {
        content: "✨ ";
    }

    /* ---------- Rank badge ---------- */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 999px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .rank-badge.top {
        background: var(--gold);
        color: #050507;
        box-shadow: 0 0 14px rgba(212, 168, 37, 0.4);
    }
    .rank-badge.other {
        background: #0e0e12;
        color: var(--gold);
        border: 1px solid rgba(212, 168, 37, 0.25);
    }

    /* ---------- Rating pill ---------- */
    .rating-pill {
        display: inline-flex;
        align-items: center;
        gap: 3px;
        background: rgba(212, 168, 37, 0.1);
        border: 1px solid rgba(212, 168, 37, 0.2);
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        color: var(--gold);
    }

    /* ---------- Directions button ---------- */
    .directions-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        width: 100%;
        padding: 8px 0;
        margin-top: auto; /* Crucial: pushes button to the absolute bottom of the card */
        background: var(--primary);
        color: #fff;
        border: 1px solid rgba(212, 168, 37, 0.15);
        border-radius: 10px;
        font-size: 13px;
        font-weight: 600;
        text-decoration: none;
        transition: opacity 0.2s;
    }
    .directions-btn:hover {
        opacity: 0.88;
        color: #fff;
    }

    /* ---------- AI Summary banner ---------- */
    .ai-summary-banner {
        background: var(--bg);
        border: 1px solid rgba(212, 168, 37, 0.15);
        border-radius: 12px;
        padding: 14px 18px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 12px;
        font-size: 14px;
        font-weight: 500;
        color: var(--text);
        line-height: 1.55;
    }

    /* ---------- Active filter chips ---------- */
    .filter-chip {
        font-size: 12px;
        font-weight: 500;
        color: var(--gold);
        background: var(--bg);
        border: 1px solid rgba(212, 168, 37, 0.2);
        padding: 5px 14px;
        border-radius: 999px;
        white-space: nowrap;
    }

    /* ---------- Preset cards ---------- */
    .preset-card {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        transition: border-color 0.2s, background 0.2s;
        cursor: pointer;
    }
    .preset-card:hover {
        border-color: rgba(212, 168, 37, 0.35);
        background: var(--surface);
    }

    /* ---------- Section dividers ---------- */
    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 8px 0;
    }

    /* ---------- Streamlit widget overrides ---------- */
    .stSelectbox label, .stSlider label, .stTextInput label, .stRadio label {
        font-weight: 600 !important;
        font-size: 13px !important;
        color: var(--text) !important;
    }
    div[data-baseweb="select"] > div {
        background-color: var(--bg) !important;
        border-color: var(--border) !important;
    }
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
    }
    
    /* ---------- Stale element fade fix ---------- */
    .element-container { transition: opacity 0.15s; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 6. SERVICE INITIALIZATION (cached — survives Streamlit reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI engine — models, embeddings & data…")
def init_services():
    """Initialize the service registry once per server process."""
    registry = ServiceRegistry()
    registry.initialize()
    return registry


svc = init_services()

# ---------------------------------------------------------------------------
# 7. METADATA HELPERS (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_locations():
    """Extract sorted unique locations from the loaded DataFrame."""
    if svc.repo is None or svc.repo._df is None:
        return []
    df = svc.repo._df
    locations = set(str(loc).strip() for loc in df["location"].dropna().unique() if str(loc).strip())
    # Enrich with common Bangalore areas that may be missing
    for extra in [
        "HSR Layout", "Whitefield", "Marathahalli", "Electronic City",
        "Basavanagudi", "Kalyan Nagar", "Richmond Town", "Ulsoor",
        "Sadashivanagar", "Hebbal", "Domlur", "Bannerghatta Road", "Kammanahalli",
    ]:
        locations.add(extra)
    return sorted(locations)


@st.cache_data(show_spinner=False)
def get_cuisines():
    """Extract sorted unique cuisines from the loaded DataFrame."""
    if svc.repo is None or svc.repo._df is None:
        return []
    df = svc.repo._df
    cuisines = set()
    for c_list in df["cuisines"].dropna():
        if isinstance(c_list, list):
            for c in c_list:
                cuisines.add(str(c).strip())
        else:
            cuisines.add(str(c_list).strip())
    for extra in [
        "North Indian", "South Indian", "Chinese", "Italian", "Continental",
        "Desserts", "Cafe", "Biryani", "Fast Food", "Mughlai", "Street Food", "Beverages",
    ]:
        cuisines.add(extra)
    return sorted(cuisines)


# ---------------------------------------------------------------------------
# 8. CUISINE IMAGE HELPER (mirrors React getCuisineImage)
# ---------------------------------------------------------------------------
def get_cuisine_image(cuisines: list[str] = []) -> str:
    lowered = [c.lower() for c in cuisines]
    if "pizza" in lowered or "italian" in lowered:
        return "https://lh3.googleusercontent.com/aida-public/AB6AXuA55ozOycurOiy5kVz0amfXraAEVUxZFTIWhLRrL5_0kb3c94npYBW6fXUlHeu59NgrK3Ifr8uN_14Mj0BuzAsA5wGPmxqB4tbo1RAr3ZF1nnOSOr7AoqtLay1plc8pyRm68J5Cwzv7rN0hRNJ-AENrJyGkzyQzjvV703PpVtVpt2vGrUN1XZXxrVCBwFGY0kh1b-G40NH5v6FpL08uFJ0UYrQWl9Wt7rHi9bKrL4kdK8ZMXOjRrc0cVEau5M03JVWpScCYGM-Hu2s"
    if "cafe" in lowered or "desserts" in lowered or "ice cream" in lowered:
        return "https://lh3.googleusercontent.com/aida-public/AB6AXuAzwM4hLxcvHB3_HSuqxoMkAl_IK4ptjITsrrzvjeeKfIbgdRifgCJGkR7V-Oyepn0sTOJ5eNL6c7PQjpZD_JWazbedt6fDOZTtgVPS01vOFZK0cg1t62uN9UBdXMR0_LQPpyvfmrWtv-ec3PQNHXVwyuCJpPt_z09GboCdewPA7wEyax7zQf4olBnpugbH_YGHyPbn4eqLaOTT2SW8ykVczVrNI3lK1LulKaONxW2G_s2IeGX-zF8vQ17-H39GjoogLFfyX8towH0"
    return "https://lh3.googleusercontent.com/aida-public/AB6AXuDFxVJLcQ30mUJaCeRl1BQOfAs8rkLiKae5K4X7U7dSnwFZ6B-syvr4XJka6HLwPxhDQoizExKgRdC5GfAwwVmQ_7DmsLGQdea6dtImYYd84SvKuMZoY7HQMVpsokE-IqhGJixen0m0rRpG1i7hC6nq2Omq5uymM9Klm9A_JNlorBfaQQgsY2ScbiPNm6l8dS2JF6I0hqZ2zRzMF3HQDIycClLJSAl1Ls7IH0V-wrH8ewteEdEv94PCrIxRbxQuByx3eSgKc03Qy5Y"


# ---------------------------------------------------------------------------
# 9. RENDER FUNCTIONS
# ---------------------------------------------------------------------------
def render_header():
    """Top branding header."""
    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            "<h1 style='color:#8B0000; font-size:36px; margin:0; padding:0;'>TasteFinder AI</h1>",
            unsafe_allow_html=True,
        )
    with right:
        loc_display = st.session_state.get("pref_location", "Bangalore")
        st.markdown(
            f"<div style='text-align:right; padding-top:10px; color:#8B0000; font-weight:600;'>"
            f"📍 {loc_display or 'Bangalore'}</div>",
            unsafe_allow_html=True,
        )


def render_search_form():
    """Preference input form — left column of the search view."""
    locations = get_locations()
    cuisines = get_cuisines()

    # Location
    loc_idx = 0
    if "pref_location" in st.session_state and st.session_state["pref_location"] in locations:
        loc_idx = locations.index(st.session_state["pref_location"])
    location = st.selectbox(
        "📍 Where are you looking?",
        options=locations,
        index=loc_idx,
        key="widget_location",
    )

    # Cuisine
    cui_idx = 0
    if "pref_cuisine" in st.session_state and st.session_state["pref_cuisine"] in cuisines:
        cui_idx = cuisines.index(st.session_state["pref_cuisine"])
    cuisine = st.selectbox(
        "🍽️ Cuisine or Dish",
        options=cuisines,
        index=cui_idx,
        key="widget_cuisine",
    )

    # Quick cuisine pills
    st.markdown(
        "<span style='font-size:11px; font-weight:700; color:#b5b5be; letter-spacing:0.06em; text-transform:uppercase;'>"
        "🔥 Top Cuisines</span>",
        unsafe_allow_html=True,
    )
    pill_cols = st.columns(3)
    for i, c in enumerate(["North Indian", "Chinese", "South Indian"]):
        with pill_cols[i]:
            if st.button(c, key=f"pill_{c}", use_container_width=True):
                st.session_state["pref_cuisine"] = c
                st.rerun()

    # Budget
    budget_options = {"Low": "low", "Medium": "medium", "High": "high"}
    default_budget = st.session_state.get("pref_budget", "medium")
    default_idx = list(budget_options.values()).index(default_budget) if default_budget in budget_options.values() else 1
    budget_label = st.radio(
        "💰 Budget",
        options=list(budget_options.keys()),
        index=default_idx,
        horizontal=True,
        key="widget_budget",
    )
    budget = budget_options[budget_label]

    # Min Rating
    default_rating = st.session_state.get("pref_min_rating", 4.0)
    min_rating = st.slider(
        "⭐ Minimum Rating",
        min_value=3.0,
        max_value=5.0,
        value=float(default_rating),
        step=0.1,
        format="%.1f",
        key="widget_rating",
    )

    # Top N
    default_topn = st.session_state.get("pref_top_n", 5)
    top_n = st.slider(
        "🍽️ How many recommendations?",
        min_value=1,
        max_value=5,
        value=int(default_topn),
        step=1,
        key="widget_topn",
    )

    # Additional vibes
    default_vibes = st.session_state.get("pref_additional", "")
    additional = st.text_input(
        "✨ Any specific vibes?",
        value=default_vibes,
        placeholder="e.g. cozy rooftop ambience, authentic handmade pasta...",
        key="widget_additional",
    )

    return location, cuisine, budget, min_rating, top_n, additional


def render_result_card(rec_detail, rank: int, is_top: bool):
    """Render a single restaurant recommendation card as styled HTML."""
    res = rec_detail["restaurant"]
    explanation = rec_detail["explanation"]

    cuisines_list = res.cuisines if isinstance(res.cuisines, list) else [str(res.cuisines)]
    img_url = get_cuisine_image(cuisines_list)
    cuisine_display = cuisines_list[0] if cuisines_list else "—"
    maps_url = f"https://www.google.com/maps/search/?api=1&query={res.name}+{res.location}".replace(" ", "+")

    badge_class = "top" if is_top else "other"
    badge_text = "🏆 #1 TOP MATCH" if is_top else f"⭐ MATCH #{rank}"

    card_html = f"""
    <div class="rec-card">
        <img class="rec-card-img" src="{img_url}" alt="{res.name}" />
        <div class="rec-card-body">
            <span class="rank-badge {badge_class}">{badge_text}</span>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <p class="rec-card-title" title="{res.name}">{res.name}</p>
                <span class="rating-pill">{res.rating:.1f} ⭐</span>
            </div>
            <div class="rec-card-meta">
                <span class="rec-card-chip">💰 ₹{int(res.estimated_cost)}</span>
                <span class="rec-card-chip">📍 {res.location}</span>
                <span class="rec-card-chip">🍽️ {cuisine_display}</span>
            </div>
            <div class="rec-card-explanation">{explanation}</div>
            <a href="{maps_url}" target="_blank" class="directions-btn">📍 Get Directions</a>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_results(summary, rec_details, prefs):
    """Render the full results view: active filters, AI summary, and cards."""

    # Active filter chips
    chips_html = "".join([
        f'<span class="filter-chip">{label}</span>'
        for label in [
            prefs["location"],
            prefs["cuisine"],
            f'{prefs["budget"].upper()} Budget',
            f'★ {prefs["min_rating"]:.1f}+',
            f'🍽️ {prefs["top_n"]} Match{"es" if prefs["top_n"] > 1 else ""}',
        ]
    ])
    st.markdown(
        f'<div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:8px;">'
        f'<span style="font-size:12px; color:#b5b5be; font-weight:500;">Active Filters:</span>'
        f'{chips_html}</div>',
        unsafe_allow_html=True,
    )

    # AI Summary
    if summary:
        st.markdown(
            f'<div class="ai-summary-banner">'
            f'<span style="font-size:22px; flex-shrink:0;">✨</span>'
            f'<span>{summary}</span></div>',
            unsafe_allow_html=True,
        )

    # Recommendation cards in a responsive grid
    count = len(rec_details)
    if count == 0:
        return

    cols = st.columns(min(count, 5))
    for idx, detail in enumerate(rec_details):
        with cols[idx % len(cols)]:
            render_result_card(detail, rank=idx + 1, is_top=(idx == 0))


def render_presets():
    """AI Quick Cravings preset buttons."""
    st.markdown(
        "<div style='margin-top:8px;'>"
        "<span style='font-size:11px; font-weight:700; color:#d4a825; letter-spacing:0.06em; text-transform:uppercase;'>"
        "💡 AI Quick Cravings Presets</span></div>",
        unsafe_allow_html=True,
    )

    presets = [
        ("🌹", "Date Night", "Italian", "high", 4.2, "High-end elegant setting, romantic outdoor vibe, delicious lasagna."),
        ("☕", "Work & Coffee", "Cafe", "medium", 4.0, "Cozy study spot, fantastic filter coffee, fresh croissants."),
        ("🍧", "Late Sweet", "Desserts", "low", 4.0, "Gelato or waffle spot, late night."),
    ]

    preset_cols = st.columns(len(presets))
    for i, (icon, name, cuisine, budget, rating, vibes) in enumerate(presets):
        with preset_cols[i]:
            card_html = f"""
            <div class="preset-card">
                <div style="font-size:24px; margin-bottom:4px;">{icon}</div>
                <div style="font-weight:600; color:#f0ece8; font-size:14px;">{name}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            if st.button(f"Apply {name}", key=f"preset_{name}", use_container_width=True):
                st.session_state["pref_location"] = "Indiranagar"
                st.session_state["pref_cuisine"] = cuisine
                st.session_state["pref_budget"] = budget
                st.session_state["pref_min_rating"] = rating
                st.session_state["pref_top_n"] = 3
                st.session_state["pref_additional"] = vibes
                st.session_state["trigger_search"] = True
                st.rerun()


# ---------------------------------------------------------------------------
# 10. CORE SEARCH LOGIC
# ---------------------------------------------------------------------------
def run_search(location, cuisine, budget, min_rating, top_n, additional):
    """Execute the recommendation pipeline and return results."""
    prefs = UserPreferences(
        location=location,
        cuisine=cuisine,
        budget=budget,
        min_rating=min_rating,
        additional_preferences=additional or None,
        top_n=top_n,
    )

    # Run orchestrator
    response = svc.orchestrator.execute(prefs)

    if not response.recommendations:
        return None, [], prefs

    # Enrich with restaurant details (mirrors routes.py logic)
    rec_ids = [rec.restaurant_id for rec in response.recommendations]
    db_restaurants = svc.repo.get_by_ids(rec_ids)
    db_lookup = {r.id: r for r in db_restaurants}

    rec_details = []
    for rec in response.recommendations:
        res = db_lookup.get(rec.restaurant_id)
        if res:
            rec_details.append({
                "rank": rec.rank,
                "restaurant": res,
                "explanation": rec.explanation,
            })

    return response.summary, rec_details, prefs


# ---------------------------------------------------------------------------
# 11. MAIN APP LAYOUT
# ---------------------------------------------------------------------------
def main():
    # Initialise session state defaults
    defaults = {
        "screen": "SEARCH",
        "pref_location": "",
        "pref_cuisine": "",
        "pref_budget": "medium",
        "pref_min_rating": 4.0,
        "pref_top_n": 5,
        "pref_additional": "",
        "results_summary": "",
        "results_details": [],
        "results_prefs": {},
        "trigger_search": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Guard: services must be healthy
    if svc.orchestrator is None or svc.repo is None:
        st.error("⚠️ Discovery engine failed to initialise. Check data ingestion and API keys.")
        st.stop()

    render_header()
    st.markdown("<hr>", unsafe_allow_html=True)

    # ----- SEARCH VIEW -----
    if st.session_state["screen"] == "SEARCH":
        left_col, right_col = st.columns([5, 7], gap="large")

        # Left — hero text
        with left_col:
            st.markdown(
                "<h2 style='font-size:32px; line-height:1.25; margin-bottom:12px;'>"
                "Better food decisions powered by "
                "<span style='color:#d4a825;'>AI</span></h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='color:#b5b5be; font-size:14px; line-height:1.7;'>"
                "Provide your craving, location, quality constraints, and unique vibe. "
                "Our engine vectorizes your search locally and orchestrates Groq AI to "
                "deliver grounded, explainable recommendations without scrolling.</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='display:flex; align-items:center; gap:10px; color:#d4a825; "
                "margin-top:16px; background:#0e1111; border:1px solid rgba(212,168,37,0.2); "
                "padding:12px 16px; border-radius:12px; max-width:300px;'>"
                "<span style='font-size:20px;'>✨</span>"
                "<div>"
                "<div style='font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;'>Grounded Vector Search</div>"
                "<div style='font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:#b5b5be; margin-top:2px;'>Strict Viewport Bounds</div>"
                "</div></div>",
                unsafe_allow_html=True,
            )

        # Right — form
        with right_col:
            location, cuisine, budget, min_rating, top_n, additional = render_search_form()

            # Submit button
            if st.button("✨ Get AI Recommendations", type="primary", use_container_width=True, key="btn_search"):
                if not location:
                    st.warning("Please select a location.")
                elif not cuisine:
                    st.warning("Please select a cuisine.")
                else:
                    # Store prefs and trigger search
                    st.session_state["pref_location"] = location
                    st.session_state["pref_cuisine"] = cuisine
                    st.session_state["pref_budget"] = budget
                    st.session_state["pref_min_rating"] = min_rating
                    st.session_state["pref_top_n"] = top_n
                    st.session_state["pref_additional"] = additional
                    st.session_state["trigger_search"] = True
                    st.rerun()

            render_presets()

    # ----- RESULTS VIEW -----
    elif st.session_state["screen"] == "RESULTS":
        col_back, _ = st.columns([1, 5])
        with col_back:
            if st.button("← Edit Search", key="btn_back"):
                st.session_state["screen"] = "SEARCH"
                st.rerun()

        render_results(
            st.session_state["results_summary"],
            st.session_state["results_details"],
            st.session_state["results_prefs"],
        )

    # ----- EMPTY STATE -----
    elif st.session_state["screen"] == "EMPTY":
        st.markdown(
            "<div style='text-align:center; padding:80px 0;'>"
            "<div style='font-size:48px; margin-bottom:12px;'>🔍</div>"
            "<h2 style='color:#f0ece8;'>No matching restaurants</h2>"
            "<p style='color:#b5b5be; max-width:400px; margin:8px auto 24px;'>"
            "Try broadening your location area, reducing rating caps, or adjusting budget constraints.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("🔧 Adjust Filters", key="btn_adjust"):
            st.session_state["screen"] = "SEARCH"
            st.rerun()

    # ----- ERROR STATE -----
    elif st.session_state["screen"] == "ERROR":
        st.markdown(
            "<div style='text-align:center; padding:80px 0;'>"
            "<div style='font-size:48px; margin-bottom:12px;'>⚠️</div>"
            "<h2 style='color:#f0ece8;'>Something went wrong</h2>"
            "<p style='color:#b5b5be; max-width:400px; margin:8px auto 24px;'>"
            "We encountered an issue during recommendation analysis. "
            "Please check your connection or LLM model tokens.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("🔄 Try Again", key="btn_retry"):
            st.session_state["screen"] = "SEARCH"
            st.rerun()

    # ----- TRIGGER SEARCH (runs after rerun) -----
    if st.session_state.get("trigger_search"):
        st.session_state["trigger_search"] = False
        with st.spinner("🔍 Finding matching restaurants… Analyzing ratings… Running vector search… AI is ranking your best options…"):
            try:
                summary, details, prefs_obj = run_search(
                    st.session_state["pref_location"],
                    st.session_state["pref_cuisine"],
                    st.session_state["pref_budget"],
                    st.session_state["pref_min_rating"],
                    st.session_state["pref_top_n"],
                    st.session_state["pref_additional"],
                )
                if not details:
                    st.session_state["screen"] = "EMPTY"
                else:
                    st.session_state["results_summary"] = summary
                    st.session_state["results_details"] = details
                    st.session_state["results_prefs"] = {
                        "location": st.session_state["pref_location"],
                        "cuisine": st.session_state["pref_cuisine"],
                        "budget": st.session_state["pref_budget"],
                        "min_rating": st.session_state["pref_min_rating"],
                        "top_n": st.session_state["pref_top_n"],
                    }
                    st.session_state["screen"] = "RESULTS"
            except Exception as e:
                traceback.print_exc()
                st.session_state["screen"] = "ERROR"
        st.rerun()

    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='display:flex; justify-content:space-between; align-items:center; "
        "flex-wrap:wrap; gap:12px; padding:4px 0;'>"
        "<span style='font-weight:700; font-size:15px; color:#f0ece8;'>TasteFinder AI</span>"
        "<span style='font-size:11px; color:#b5b5be;'>© 2026 TasteFinder AI by Zomato. All rights reserved.</span>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
