import random

import streamlit as st

from book_finder_engine import (
    load_books,
    add_reader_moods,
    add_text_analysis,
    load_association_rules,
    make_preferences,
    recommend_books,
)


# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Book Finder",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------

st.markdown(
    """
<style>

:root {
    --bg: #F7F3EC;
    --panel: #FFFFFF;
    --panel-soft: #FCFAF6;
    --border: #E6DDD1;
    --text: #1F2933;
    --muted: #6B7280;
    --primary: #8B4513;
    --primary-dark: #6F3810;
    --primary-soft: #F3E8DD;
    --secondary: #C99255;
    --secondary-soft: #FBF4EC;
    --success: #3F6B56;
    --info: #355C7D;
    --gold: #B07A2A;
    --shadow: 0 12px 28px rgba(60, 40, 20, 0.08);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(201,146,85,0.12), transparent 22%),
        radial-gradient(circle at top right, rgba(139,69,19,0.06), transparent 20%),
        linear-gradient(180deg, #FBF8F3 0%, var(--bg) 100%);
    color: var(--text);
}

.block-container {
    max-width: 1320px;
    padding-top: 1.1rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

html, body, [class*="css"] {
    font-size: 17px;
}

body {
    color: var(--text);
}

p, li, label {
    color: var(--text);
    line-height: 1.6;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
.stCaption,
label {
    font-size: 1rem;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 3.15rem !important;
    font-weight: 700 !important;
}

h2 {
    font-size: 2.1rem !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
    font-weight: 700 !important;
}

h3 {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}

/* Hero */
.hero-panel {
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(252,250,246,0.98));
    border: 1px solid var(--border);
    border-radius: 24px;
    box-shadow: var(--shadow);
    padding: 2.1rem 2rem 1.85rem 2rem;
    text-align: center;
    margin-bottom: 0.55rem;
}

.hero-icon {
    font-size: 2.8rem;
    line-height: 1;
    margin-bottom: 0.35rem;
}

.hero-title {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 3.55rem;
    line-height: 1.02;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    color: var(--primary);
    font-size: 1.32rem;
    font-weight: 700;
    margin-bottom: 0.55rem;
}

.hero-copy {
    color: var(--muted);
    font-size: 1.04rem;
    max-width: 820px;
    margin: 0 auto;
    line-height: 1.65;
}

/* Panels and containers */
div[data-testid="stForm"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel);
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    box-shadow: var(--shadow);
}

div[data-testid="stForm"] {
    padding: 0.95rem 1.05rem 1.1rem 1.05rem;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    border-radius: 12px !important;
    background: #FFFFFF !important;
    border: 1px solid #D7CCBD !important;
    min-height: 46px;
}

textarea, input {
    color: var(--text) !important;
}

[data-baseweb="tag"] {
    background: var(--secondary-soft) !important;
    border-radius: 999px !important;
}

/* Sliders */
[data-baseweb="slider"] [role="slider"] {
    background: var(--primary) !important;
}

/* Buttons */
div.stButton > button,
div.stFormSubmitButton > button {
    border-radius: 14px;
    font-weight: 700;
    min-height: 3.85rem;
    font-size: 1.07rem;
    border: 1px solid transparent;
    box-shadow: none;
    transition: all 0.16s ease;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 20px rgba(80, 50, 20, 0.12);
}

button[kind="primary"] {
    background: linear-gradient(180deg, #9A4E19, #7E3E12) !important;
    color: #FFFFFF !important;
    border-color: #7E3E12 !important;
}

button[kind="primary"] * {
    color: #FFFFFF !important;
}

button[kind="primary"] p,
button[kind="primary"] span,
button[kind="primary"] div {
    color: #FFFFFF !important;
}

button[kind="secondary"] {
    background: #FFFFFF !important;
    color: var(--text) !important;
    border: 1px solid #D9CBB9 !important;
}

/* Specific large choice buttons */
button[kind="primary"] p,
button[kind="secondary"] p {
    font-weight: 700 !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: var(--panel-soft);
    border: 1px solid #EEE4D8;
    border-radius: 14px;
    padding: 0.9rem 1rem;
}

div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.98rem !important;
}

div[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 2rem;
    font-weight: 700;
}

.stCaption {
    color: var(--muted) !important;
    opacity: 1;
}

/* Result card pieces */
.book-rank,
.surprise-label {
    display: inline-block;
    background: var(--primary-soft);
    color: var(--primary);
    font-size: 0.84rem;
    font-weight: 800;
    padding: 0.34rem 0.72rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
    border: 1px solid #E7D4C0;
}

.book-author {
    color: var(--muted);
    font-size: 1.02rem;
    margin-bottom: 0.85rem;
}

.book-chip {
    display: inline-block;
    background: #F5EFE6;
    color: #5E4832;
    border: 1px solid #E4D7C8;
    padding: 0.3rem 0.68rem;
    border-radius: 999px;
    font-size: 0.86rem;
    margin-right: 0.38rem;
    margin-bottom: 0.38rem;
}

.mood-chip {
    display: inline-block;
    background: #FBF4EC;
    color: #7E4B1C;
    border: 1px solid #EDD7BC;
    padding: 0.3rem 0.68rem;
    border-radius: 999px;
    font-size: 0.86rem;
    margin-right: 0.38rem;
    margin-bottom: 0.38rem;
}

.section-note {
    background: var(--secondary-soft);
    border: 1px solid #E9D7C3;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    color: #5B4A38;
}

.description-box {
    background: #FCFAF6;
    border: 1px solid #EFE5DA;
    border-radius: 14px;
    padding: 1rem;
}

/* Expanders and alerts */
details {
    background: #FCFAF6 !important;
    border: 1px solid #E6DDD1 !important;
    border-radius: 14px !important;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

/* Divider */
hr {
    border-color: rgba(80, 60, 40, 0.08) !important;
}

/* Footer */
.book-finder-footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.96rem;
    padding-top: 0.6rem;
}

/* Reduce space between blocks */
div[data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}

@media (max-width: 900px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero-title {
        font-size: 2.5rem;
    }

    h1 {
        font-size: 2.45rem !important;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# PREPARE RECOMMENDATION ENGINE
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def prepare_engine():
    """Load and prepare the expensive recommendation engine once."""

    books = load_books()
    books = add_reader_moods(books)
    books = add_text_analysis(books)
    rules = load_association_rules()

    return books, rules


# ---------------------------------------------------------
# SESSION STATE HELPERS
# ---------------------------------------------------------

RESULT_STATE_KEYS = [
    "results",
    "relaxation_notes",
    "preferences",
    "reading_history",
    "favorite_warning",
    "surprise_index",
]

FORM_STATE_KEYS = [
    "genre",
    "length",
    "minimum_rating",
    "mood",
    "avoid_genres",
    "favorite_1",
    "favorite_2",
    "favorite_3",
    "favorite_4",
    "favorite_5",
]


def initialize_state():
    """Create session-state values used by the app."""

    if "results" not in st.session_state:
        st.session_state.results = None

    if "relaxation_notes" not in st.session_state:
        st.session_state.relaxation_notes = []

    if "preferences" not in st.session_state:
        st.session_state.preferences = None

    if "reading_history" not in st.session_state:
        st.session_state.reading_history = []

    if "favorite_warning" not in st.session_state:
        st.session_state.favorite_warning = False

    if "surprise_index" not in st.session_state:
        st.session_state.surprise_index = None

    if "recommendation_mode" not in st.session_state:
        st.session_state.recommendation_mode = "discover"


def clear_results():
    """Clear old recommendations when the recommendation mode changes."""

    st.session_state.results = None
    st.session_state.relaxation_notes = []
    st.session_state.preferences = None
    st.session_state.reading_history = []
    st.session_state.favorite_warning = False
    st.session_state.surprise_index = None


def reset_app():
    """Reset the form and current recommendations without rebuilding the cache."""

    for key in FORM_STATE_KEYS + RESULT_STATE_KEYS:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.recommendation_mode = "discover"


def set_recommendation_mode(mode):
    """Switch recommendation paths and clear stale results."""

    if st.session_state.get("recommendation_mode") != mode:
        st.session_state.recommendation_mode = mode
        clear_results()


initialize_state()


# ---------------------------------------------------------
# HELPER RENDER FUNCTIONS
# ---------------------------------------------------------

def render_cover(book, height=240):
    if "image_url" in book.index and str(book["image_url"]).startswith("http"):
        st.image(book["image_url"], use_container_width=True)
    else:
        st.html(
            (
                f'<div style="height:{height}px; border-radius:14px; '
                'background:#F5EFE6; border:1px solid #E7DCCE; '
                'display:flex; align-items:center; justify-content:center; '
                'font-size:2.7rem; color:#8B4513;">📚</div>'
            )
        )


def render_badges(items, css_class, limit=None):
    if not items:
        return

    values = items if limit is None else items[:limit]
    html = ""

    for item in values:
        html += (
            f'<span class="{css_class}">' + str(item).replace("-", " ").title() + "</span>"
        )

    st.html(html)


# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------

hero_html = (
    '<div class="hero-panel">'
    '<div class="hero-icon">📚</div>'
    '<div class="hero-title">Book Finder</div>'
    '<div class="hero-subtitle">Your next great read is hiding somewhere in 9,680 books.</div>'
    '<div class="hero-copy">'
    "Tell us what you\'re in the mood for. Book Finder combines reader preferences, Goodreads data, text analytics, and reading patterns to create a personalized shortlist."
    '</div>'
    '</div>'
)

st.html(hero_html)
st.divider()


# ---------------------------------------------------------
# RECOMMENDATION STYLE
# ---------------------------------------------------------

st.header("How would you like to find your next book?")
st.write("Choose the path that fits you best. You can switch anytime.")

mode = st.session_state.recommendation_mode
mode_col1, mode_col2 = st.columns(2, gap="large")

with mode_col1:
    st.button(
        "🔎  Discover Something New",
        key="mode_discover",
        type="primary" if mode == "discover" else "secondary",
        use_container_width=True,
        on_click=set_recommendation_mode,
        args=("discover",),
    )
    st.caption("Use your genre, mood, length, and rating preferences.")

with mode_col2:
    st.button(
        "❤️  Recommend From Books I Loved",
        key="mode_loved",
        type="primary" if mode == "loved" else "secondary",
        use_container_width=True,
        on_click=set_recommendation_mode,
        args=("loved",),
    )
    st.caption("Blend your favorite books with your other preferences.")

recommendation_mode = st.session_state.recommendation_mode
st.divider()


# ---------------------------------------------------------
# PREFERENCE FORM
# ---------------------------------------------------------

with st.form("book_preferences"):
    st.header("1. What are you in the mood for?")

    col1, col2 = st.columns(2)

    with col1:
        genre = st.selectbox(
            "Preferred genre",
            [
                "Any",
                "Fantasy",
                "Romance",
                "Mystery",
                "Thriller",
                "Historical Fiction",
                "Science Fiction",
                "Young Adult",
                "Classics",
                "Horror",
                "Crime",
                "Biography",
                "Memoir",
                "History",
                "Business",
                "Psychology",
                "Philosophy",
                "Poetry",
                "Travel",
            ],
            key="genre",
        )

    with col2:
        length = st.selectbox(
            "Preferred book length",
            ["Any", "Short", "Medium", "Long"],
            key="length",
        )

    minimum_rating = st.slider(
        "Minimum Goodreads rating",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
        help="Choose 0 if rating does not matter to you.",
        key="minimum_rating",
    )

    mood = st.text_input(
        "Describe the mood or type of story you want",
        placeholder="Example: dark psychological mystery with unexpected twists",
        key="mood",
    )

    avoid_genres = st.multiselect(
        "Anything you want to avoid?",
        [
            "Fantasy",
            "Romance",
            "Mystery",
            "Thriller",
            "Historical Fiction",
            "Science Fiction",
            "Young Adult",
            "Classics",
            "Horror",
            "Crime",
            "Biography",
            "Memoir",
            "History",
        ],
        help="Optional",
        key="avoid_genres",
    )

    loved_books = []

    if recommendation_mode == "loved":
        st.divider()
        st.header("2. Tell us what you've loved")
        st.caption(
            "Enter up to five books you've enjoyed. Well-known books work best because they have more reader-rating data."
        )

        favorite_1 = st.text_input(
            "Favorite book 1",
            placeholder="Example: Gone Girl",
            key="favorite_1",
        )
        favorite_2 = st.text_input(
            "Favorite book 2",
            placeholder="Example: Sharp Objects",
            key="favorite_2",
        )
        favorite_3 = st.text_input(
            "Favorite book 3",
            placeholder="Optional",
            key="favorite_3",
        )
        favorite_4 = st.text_input(
            "Favorite book 4",
            placeholder="Optional",
            key="favorite_4",
        )
        favorite_5 = st.text_input(
            "Favorite book 5",
            placeholder="Optional",
            key="favorite_5",
        )

    st.divider()

    submitted = st.form_submit_button(
        "📖 Find My Next Read",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------
# GENERATE NEW RECOMMENDATIONS
# ---------------------------------------------------------

if submitted:
    if recommendation_mode == "loved":
        for book in [favorite_1, favorite_2, favorite_3, favorite_4, favorite_5]:
            if book.strip() != "":
                loved_books.append(book.strip())

    try:
        with st.spinner("Searching thousands of books for your best matches..."):
            books, rules = prepare_engine()

            preferences = make_preferences(
                books,
                genre,
                length,
                minimum_rating,
                mood,
                loved_books,
                avoid_genres,
            )

            results, relaxation_notes = recommend_books(
                books,
                preferences,
                rules,
            )

        st.session_state.results = results
        st.session_state.relaxation_notes = relaxation_notes
        st.session_state.preferences = preferences
        st.session_state.reading_history = preferences["loved_names"]
        st.session_state.surprise_index = None

        if (
            recommendation_mode == "loved"
            and len(loved_books) > 0
            and len(preferences["loved_names"]) < len(loved_books)
        ):
            st.session_state.favorite_warning = True
        else:
            st.session_state.favorite_warning = False

    except Exception as error:
        st.session_state.results = None
        st.error(
            "Book Finder ran into an error while creating recommendations. Please try again."
        )

        with st.expander("Technical details"):
            st.code(str(error))


# ---------------------------------------------------------
# DISPLAY STORED RESULTS
# ---------------------------------------------------------

if st.session_state.results is not None:
    results = st.session_state.results
    relaxation_notes = st.session_state.relaxation_notes

    for note in relaxation_notes:
        st.info(note)

    if len(st.session_state.reading_history) > 0:
        st.caption(
            "Using your reading history: " + ", ".join(st.session_state.reading_history)
        )

    if st.session_state.favorite_warning:
        st.warning(
            "Some favorite books could not be matched to the catalog. Recommendations were created using the books that were recognized."
        )

    st.divider()
    st.header("✨ Your Top 5 Books")
    st.write(
        "These recommendations were selected from the Book Finder catalog based on your preferences and reading patterns."
    )

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        surprise_clicked = st.button(
            "🎲 Surprise Me",
            use_container_width=True,
            disabled=(len(results) == 0),
            help="Randomly spotlight one book from your personalized Top 5.",
        )

    with action_col2:
        st.button(
            "↻ Start Over",
            use_container_width=True,
            on_click=reset_app,
            help="Clear your preferences and begin a new recommendation search.",
        )

    st.caption(
        "Surprise Me chooses from your personalized Top 5, so it still respects your recommendation results."
    )

    if surprise_clicked and len(results) > 0:
        possible_indexes = list(range(len(results)))
        previous_index = st.session_state.surprise_index

        if len(possible_indexes) > 1 and previous_index in possible_indexes:
            possible_indexes.remove(previous_index)

        st.session_state.surprise_index = random.choice(possible_indexes)

    if st.session_state.surprise_index is not None and len(results) > 0:
        surprise_book = results.iloc[st.session_state.surprise_index]

        st.subheader("🎲 Your Surprise Pick")

        with st.container(border=True):
            surprise_cover_col, surprise_info_col = st.columns([1, 4], vertical_alignment="top")

            with surprise_cover_col:
                render_cover(surprise_book, height=240)

            with surprise_info_col:
                st.html('<div class="surprise-label">SURPRISE PICK</div>')
                st.subheader(surprise_book["title"])
                st.html('<div class="book-author">by ' + str(surprise_book["author"]) + '</div>')

                stat1, stat2, stat3 = st.columns(3)
                with stat1:
                    st.metric("⭐ Rating", str(round(surprise_book["average_rating"], 2)))
                with stat2:
                    st.metric("📖 Pages", str(int(surprise_book["pages"])))
                with stat3:
                    st.metric("🎯 Match Score", str(round(surprise_book["score"], 2)))

                surprise_reasons = [
                    reason.capitalize()
                    for reason in str(surprise_book["why"]).split("; ")
                    if reason.strip() != ""
                ]

                if surprise_reasons:
                    st.markdown(
                        "<div class='section-note'><strong>Why it fits:</strong> "
                        + " • ".join(surprise_reasons[:3])
                        + "</div>",
                        unsafe_allow_html=True,
                    )

        st.divider()

    if len(results) == 0:
        st.error(
            "Book Finder could not find a suitable recommendation. Try changing one of your preferences."
        )
    else:
        for number in range(len(results)):
            book = results.iloc[number]

            with st.container(border=True):
                cover_col, info_col = st.columns([1, 4], vertical_alignment="top")

                with cover_col:
                    render_cover(book, height=230)

                with info_col:
                    st.html('<div class="book-rank">' + "PICK #" + str(number + 1) + "</div>")
                    st.subheader(book["title"])
                    st.html('<div class="book-author">by ' + str(book["author"]) + '</div>')

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⭐ Rating", str(round(book["average_rating"], 2)))
                    with col2:
                        st.metric("📖 Pages", str(int(book["pages"])))
                    with col3:
                        st.metric("🎯 Match Score", str(round(book["score"], 2)))

                    if len(book["genres_list"]) > 0:
                        st.write("**Genres**")
                        render_badges(book["genres_list"], "book-chip", limit=6)

                    if len(book["moods_list"]) > 0:
                        st.write("**Reader moods**")
                        render_badges(book["moods_list"], "mood-chip")

                    with st.expander("✨ Why Book Finder picked this book"):
                        reasons = [
                            reason.capitalize()
                            for reason in str(book["why"]).split("; ")
                            if reason.strip() != ""
                        ]

                        for reason in reasons:
                            st.write("✓ " + reason)

                        tone_col, spacer_col = st.columns([1, 3])
                        with tone_col:
                            st.caption("Text-analysis tone: " + str(book["tone"]).capitalize())

                        if "description" in book.index and str(book["description"]).strip() != "":
                            st.write("**About the book**")
                            description_text = str(book["description"]).strip()
                            if len(description_text) > 650:
                                description_text = description_text[:650] + "..."
                            st.markdown(
                                "<div class='description-box'>" + description_text + "</div>",
                                unsafe_allow_html=True,
                            )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()
st.html(
    '<div class="book-finder-footer">Built with Python, Pandas, NLTK, association-rule mining, Goodreads data, and Streamlit.</div>'
)
