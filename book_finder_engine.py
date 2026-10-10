import os
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# ---------------------------------------------------------
# NLTK SETUP
# ---------------------------------------------------------

needed = {
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "averaged_perceptron_tagger_eng":
        "taggers/averaged_perceptron_tagger_eng",
    "vader_lexicon": "sentiment/vader_lexicon.zip"
}

for name, path in needed.items():

    try:
        nltk.data.find(path)

    except LookupError:
        nltk.download(name)


# ---------------------------------------------------------
# DATA SOURCE
# ---------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "malcolmosh/goodbooks-10k/master/"
)


def read_data(file_name):

    # Use a local copy if one exists
    if os.path.exists(file_name):
        return pd.read_csv(file_name)

    # Otherwise download the file from GitHub
    return pd.read_csv(DATA_URL + file_name)


# ---------------------------------------------------------
# DATA CLEANING HELPERS
# ---------------------------------------------------------

def make_list(value):

    if pd.isna(value):
        return []

    text = (
        str(value)
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )

    items = text.split(",")

    clean_items = []

    for item in items:

        item = " ".join(item.split())

        if item != "":
            clean_items.append(item)

    return clean_items


def is_sequel(title):

    lower_title = title.lower()

    if (
        "box set" in lower_title
        or "boxed set" in lower_title
        or "boxset" in lower_title
    ):
        return True

    if "#" not in title:
        return False

    if "#1)" in title:
        return False

    return True


# ---------------------------------------------------------
# LOAD AND PREPARE BOOK DATA
# ---------------------------------------------------------

def load_books():

    books = read_data("books_enriched.csv")

    # Keep English books only
    books = books[
        books["language_code"] == "eng"
    ].copy()

    books = books.reset_index(drop=True)


    # Replace missing descriptions
    books["description"] = (
        books["description"].fillna("")
    )


    # Fix missing or impossible page counts
    median_pages = books["pages"].median()

    fixed_pages = []

    for pages in books["pages"]:

        if pd.isna(pages) or pages < 10:
            fixed_pages.append(median_pages)

        else:
            fixed_pages.append(pages)

    books["pages"] = fixed_pages


    # Convert genre text into Python lists
    books["genres_list"] = (
        books["genres"].apply(make_list)
    )


    # Remove overly general genres
    genre_lists = []

    for genre_list in books["genres_list"]:

        useful = []

        for genre in genre_list:

            if genre != "fiction" and genre != "books":
                useful.append(genre)

        genre_lists.append(useful)

    books["genres_list"] = genre_lists


    # Extract the primary author
    first_authors = []

    for text in books["authors"]:

        author_list = make_list(text)

        if len(author_list) > 0:
            first_authors.append(author_list[0])

        else:
            first_authors.append("Unknown")

    books["author"] = first_authors


    # Identify sequels and box sets
    sequels = []

    for title in books["title"]:

        sequels.append(
            is_sequel(title)
        )

    books["is_sequel"] = sequels


    return books


# ---------------------------------------------------------
# READER MOOD TAGS
# ---------------------------------------------------------

# Goodreads shelf names that represent different reader moods
MOOD_SHELVES = {

    "funny": [
        "funny",
        "humor",
        "humour",
        "hilarious",
        "comedy",
        "humorous",
        "laugh-out-loud"
    ],

    "dark": [
        "dark",
        "disturbing",
        "twisted",
        "dark-fantasy",
        "grimdark",
        "psychological-thriller"
    ],

    "cozy": [
        "cozy",
        "cozy-mystery",
        "cozy-mysteries",
        "feel-good",
        "light-reading",
        "comfort-read",
        "heartwarming"
    ],

    "emotional": [
        "made-me-cry",
        "tear-jerker",
        "tearjerker",
        "emotional",
        "sad",
        "heartbreaking"
    ],

    "scary": [
        "horror",
        "scary",
        "creepy",
        "spooky",
        "ghosts",
        "haunted"
    ],

    "romantic": [
        "romance",
        "romantic",
        "love-story",
        "chick-lit",
        "contemporary-romance"
    ],

    "thought-provoking": [
        "thought-provoking",
        "philosophy",
        "psychology",
        "social-issues",
        "life-changing"
    ],

    "adventurous": [
        "adventure",
        "action-adventure",
        "action",
        "epic-fantasy",
        "quest"
    ],

    "inspiring": [
        "inspirational",
        "inspiration",
        "inspiring",
        "motivational",
        "self-help",
        "self-improvement"
    ],

    "beach-read": [
        "beach-read",
        "beach-reads",
        "summer-reading",
        "summer-reads",
        "vacation-reads"
    ]
}


# Different moods appear on Goodreads shelves at different rates,
# so each mood uses its own minimum threshold.
MOOD_THRESHOLD = {

    "funny": 0.01,
    "dark": 0.003,
    "cozy": 0.002,
    "emotional": 0.003,
    "scary": 0.01,
    "romantic": 0.02,
    "thought-provoking": 0.005,
    "adventurous": 0.01,
    "inspiring": 0.005,
    "beach-read": 0.002
}


# ---------------------------------------------------------
# TEXT ANALYSIS HELPERS
# ---------------------------------------------------------

# NLTK stopwords plus common words people use when asking
# for a recommendation.
stop_words = stopwords.words("english")

stop_words = stop_words + [
    "want",
    "something",
    "anything",
    "like",
    "book",
    "books",
    "read",
    "really",
    "please",
    "would",
    "looking",
    "kind",
    "type",
    "get",
    "story",
    "novel",
    "one"
]


# Turn text into lowercase words and remove punctuation
# and stopwords.
def clean_words(text):

    text = str(text).lower().replace("-", " ")

    tokens = word_tokenize(text)

    words = []

    for token in tokens:

        if token.isalpha() and token not in stop_words:
            words.append(token)

    return words


# Convert a VADER compound sentiment score into
# an easy-to-understand tone category.
def find_tone(score):

    if score >= 0.5:
        return "light"

    elif score <= -0.5:
        return "heavy"

    else:
        return "balanced"

# ---------------------------------------------------------
# ADD READER MOODS TO BOOK DATA
# ---------------------------------------------------------

def add_reader_moods(books):

    tags = read_data("tags.csv")
    book_tags = read_data("book_tags.csv")


    # Connect each Goodreads shelf/tag to one of our moods
    shelf_to_mood = {}

    for mood, shelves in MOOD_SHELVES.items():

        for shelf in shelves:
            shelf_to_mood[shelf] = mood


    # Create dictionary:
    # tag_id -> tag name
    tag_names = {}

    tag_ids = tags["tag_id"].tolist()
    names = tags["tag_name"].tolist()

    for i in range(len(tag_ids)):
        tag_names[tag_ids[i]] = names[i]


    total_shelvings = {}
    mood_shelvings = {}


    shelf_books = book_tags["goodreads_book_id"].tolist()
    shelf_ids = book_tags["tag_id"].tolist()
    counts = book_tags["count"].tolist()


    # Count how often readers placed each book on each mood shelf
    for i in range(len(shelf_books)):

        book = shelf_books[i]
        count = counts[i]

        # Ignore incorrect negative counts
        if count < 0:
            continue


        total_shelvings.setdefault(book, 0)
        total_shelvings[book] += count


        shelf = tag_names[shelf_ids[i]]


        if shelf in shelf_to_mood:

            mood = shelf_to_mood[shelf]

            mood_shelvings.setdefault(book, {})

            mood_shelvings[book].setdefault(mood, 0)

            mood_shelvings[book][mood] += count


    # Assign moods to each book
    moods_lists = []


    for goodreads_id in books["goodreads_book_id"]:

        moods = []

        if goodreads_id in mood_shelvings:

            for mood, count in mood_shelvings[goodreads_id].items():

                mood_share = (
                    count /
                    total_shelvings[goodreads_id]
                )

                if mood_share >= MOOD_THRESHOLD[mood]:
                    moods.append(mood)


        moods_lists.append(moods)


    books = books.copy()

    books["moods_list"] = moods_lists


    return books



# ---------------------------------------------------------
# ANALYZE BOOK DESCRIPTIONS WITH NLTK
# ---------------------------------------------------------

def add_text_analysis(books):

    sia = SentimentIntensityAnalyzer()


    description_words = []
    keywords_lists = []
    sentiments = []
    tones = []


    for description in books["description"]:

        # Clean and tokenize the description
        words = clean_words(description)

        description_words.append(words)


        # POS tag the words and keep nouns as possible themes
        nouns = []

        for word, tag in nltk.pos_tag(words):

            if (tag == "NN" or tag == "NNS") and len(word) > 2:
                nouns.append(word)


        # Keep the eight most common nouns
        keywords = []

        frequency = FreqDist(nouns)

        for word, number in frequency.most_common(8):
            keywords.append(word)

        keywords_lists.append(keywords)


        # Calculate sentiment of the original description
        sentiment_score = (
            sia.polarity_scores(description)["compound"]
        )

        sentiments.append(sentiment_score)

        tones.append(
            find_tone(sentiment_score)
        )


    books = books.copy()

    books["description_words"] = description_words
    books["keywords_list"] = keywords_lists
    books["sentiment"] = sentiments
    books["tone"] = tones


    return books

# ---------------------------------------------------------
# ASSOCIATION RULES
# "Readers who loved X also loved Y"
# ---------------------------------------------------------

def build_rules(ratings_table):

    # Treat ratings of 4 or 5 stars as books the reader loved
    loved = ratings_table[
        ratings_table["rating"] >= 4
    ]


    # Keep the 1,000 most-loved books so the analysis
    # remains manageable on a normal laptop
    top_books = (
        loved["book_id"]
        .value_counts()
        .head(1000)
        .index
        .tolist()
    )

    loved = loved[
        loved["book_id"].isin(top_books)
    ]


    # Create a reader x book basket
    # True means the reader loved that book
    basket = (
        pd.crosstab(
            loved["user_id"],
            loved["book_id"]
        ) > 0
    )


    # Find book combinations that appear together
    frequent_itemsets = apriori(
        basket,
        min_support=0.005,
        use_colnames=True,
        max_len=2,
        low_memory=True
    )


    # Build association rules
    rules_table = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=1
    )


    # Convert the rule table into a simpler dictionary:
    #
    # book_id:
    # [
    #     [other_book_id, confidence, lift],
    #     ...
    # ]

    rules = {}

    antecedents = rules_table["antecedents"].tolist()
    consequents = rules_table["consequents"].tolist()
    confidence_values = rules_table["confidence"].tolist()
    lift_values = rules_table["lift"].tolist()


    for i in range(len(antecedents)):

        book_id = list(antecedents[i])[0]

        other_book_id = list(consequents[i])[0]

        rules.setdefault(
            book_id,
            []
        )

        rules[book_id].append(
            [
                other_book_id,
                confidence_values[i],
                lift_values[i]
            ]
        )


    return rules

def load_association_rules():

    # Load the Goodreads ratings data
    ratings = read_data("ratings.csv")

    # Build the association-rule model
    rules = build_rules(ratings)

    return rules

# ---------------------------------------------------------
# USER PREFERENCE INTERPRETATION
# ---------------------------------------------------------

# Match normal user language to genres used in the catalog
genre_synonyms = {

    "fantasy": [
        "fantasy", "magic", "magical", "wizard",
        "wizards", "witch", "witches", "dragon", "dragons"
    ],

    "romance": [
        "romance", "romantic", "love story", "relationship"
    ],

    "mystery": [
        "mystery", "mysterious", "detective",
        "investigation", "whodunit"
    ],

    "thriller": [
        "thriller", "thrilling", "suspense",
        "suspenseful", "tense"
    ],

    "historical-fiction": [
        "historical fiction", "historical", "period piece"
    ],

    "nonfiction": [
        "nonfiction", "non fiction", "true story",
        "factual", "real life"
    ],

    "science-fiction": [
        "science fiction", "sci fi", "space",
        "future", "futuristic", "alien", "aliens"
    ],

    "young-adult": [
        "young adult", "ya", "teen", "teenager", "teenage"
    ],

    "classics": [
        "classic", "classics", "classic literature"
    ],

    "horror": [
        "horror", "scary", "creepy",
        "frightening", "terrifying"
    ],

    "crime": [
        "crime", "criminal", "police", "murder"
    ],

    "biography": [
        "biography", "biographical", "life story"
    ],

    "memoir": [
        "memoir", "personal story", "autobiography"
    ],

    "self-help": [
        "self help", "personal growth", "improvement"
    ],

    "graphic-novels": [
        "graphic novel", "graphic novels"
    ],

    "comics": [
        "comic", "comics", "comic book"
    ],

    "history": [
        "history", "historical nonfiction"
    ],

    "business": [
        "business", "management", "leadership",
        "entrepreneur", "entrepreneurship"
    ],

    "psychology": [
        "psychology", "behavior", "behaviour", "mind"
    ],

    "philosophy": [
        "philosophy", "philosophical"
    ],

    "poetry": [
        "poetry", "poem", "poems"
    ],

    "travel": [
        "travel", "traveling", "travelling"
    ],

    "sports": [
        "sports", "sport", "athlete", "athletics"
    ],

    "cookbooks": [
        "cookbook", "cookbooks", "cooking", "recipes"
    ],

    "paranormal": [
        "paranormal", "supernatural", "ghost",
        "ghosts", "vampire", "vampires"
    ],

    "contemporary": [
        "contemporary", "modern day", "present day"
    ]
}


# User mood words -> reader mood categories
mood_synonyms = {

    "funny": [
        "funny", "humorous", "humor", "comedy",
        "laugh", "hilarious", "lighthearted",
        "light hearted", "witty"
    ],

    "dark": [
        "dark", "twisted", "disturbing", "gritty", "bleak"
    ],

    "cozy": [
        "cozy", "cosy", "comforting", "heartwarming",
        "feel good", "wholesome", "gentle"
    ],

    "emotional": [
        "sad", "emotional", "heartbreaking",
        "cry", "tearjerker", "moving", "touching"
    ],

    "scary": [
        "scary", "creepy", "spooky", "frightening",
        "terrifying", "horror", "haunted"
    ],

    "romantic": [
        "romantic", "romance", "love story", "swoon"
    ],

    "thought-provoking": [
        "thoughtful", "thought provoking",
        "deep", "philosophical", "smart"
    ],

    "adventurous": [
        "adventurous", "adventure",
        "exciting", "epic", "quest", "action"
    ],

    "inspiring": [
        "inspiring", "inspirational",
        "uplifting", "motivating", "hopeful"
    ],

    "beach-read": [
        "beach", "vacation", "summer",
        "easy read", "light read"
    ]
}


light_words = [
    "happy",
    "cheerful",
    "lighthearted",
    "light hearted",
    "feel good",
    "uplifting",
    "hopeful",
    "fun",
    "joyful"
]


heavy_words = [
    "sad",
    "dark",
    "intense",
    "heavy",
    "serious",
    "heartbreaking",
    "bleak"
]

# Match complete words/phrases rather than pieces of words
# Example: "ya" should not match "kayak"
def has_phrase(text, phrase):

    text = " " + " ".join(clean_words(text)) + " "
    phrase = " " + " ".join(clean_words(phrase)) + " "

    return phrase in text

# Separate things the user wants from things they do not want
negative_words = [
    "not",
    "no",
    "never",
    "without",
    "n't",
    "hate",
    "avoid",
    "nothing",
    "except"
]

end_words = [
    ",",
    ".",
    ";",
    "!",
    "?",
    "but"
]


def split_wanted_unwanted(text):

    tokens = word_tokenize(
        str(text).lower()
    )

    wanted = []
    unwanted = []

    negative = False


    for i in range(len(tokens)):

        token = tokens[i]


        # Example: "anything but scary"
        if (
            token == "but"
            and i > 0
            and tokens[i - 1] in ["anything", "everything"]
        ):

            negative = True


        elif token in negative_words:

            negative = True


        elif token in end_words:

            negative = False


        elif negative:

            unwanted.append(token)


        else:

            wanted.append(token)


    return (
        " ".join(wanted),
        " ".join(unwanted)
    )

def match_genre(user_input, catalog_genres):

    if (
        user_input.strip() == ""
        or user_input.lower().strip() == "any"
    ):
        return "any"


    # First try an exact catalog genre
    for genre in catalog_genres:

        if (
            " ".join(clean_words(user_input))
            ==
            " ".join(clean_words(genre))
        ):
            return genre


    # Then try synonyms
    for genre in genre_synonyms:

        if genre in catalog_genres:

            for synonym in genre_synonyms[genre]:

                if has_phrase(user_input, synonym):
                    return genre


    return "any"

def match_moods(user_input):

    moods = []


    for mood in mood_synonyms:

        for word in mood_synonyms[mood]:

            if (
                has_phrase(user_input, word)
                and mood not in moods
            ):

                moods.append(mood)


    return moods

def match_tone(user_input):

    for word in light_words:

        if has_phrase(user_input, word):
            return "light"


    for word in heavy_words:

        if has_phrase(user_input, word):
            return "heavy"


    return "any"

def match_length(user_input):

    user_input = user_input.lower()


    if (
        "short" in user_input
        or "quick" in user_input
        or "small" in user_input
    ):

        return "short"


    elif (
        "medium" in user_input
        or "average" in user_input
        or "normal" in user_input
    ):

        return "medium"


    elif (
        "long" in user_input
        or "big" in user_input
        or "epic" in user_input
    ):

        return "long"


    # Also understand numeric inputs such as "300 pages"
    for word in user_input.split():

        if word.isdigit():

            pages = int(word)

            if pages < 300:
                return "short"

            elif pages <= 500:
                return "medium"

            else:
                return "long"


    return "any"

def title_words(text):

    text = str(text).lower()

    marks = [
        ".",
        ",",
        "!",
        "?",
        ";",
        ":",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "'",
        '"',
        "-",
        "/",
        "_",
        "#",
        "&"
    ]


    for mark in marks:
        text = text.replace(mark, " ")


    return text.split()

def find_book(books, user_title):

    user_words = title_words(
        user_title.replace("'", "")
    )


    if len(user_words) == 0:
        return None


    best_index = None


    for index, row in books.iterrows():

        words_in_title = title_words(
            row["title"].replace("'", "")
        )


        all_found = True


        for word in user_words:

            if word not in words_in_title:
                all_found = False


        if all_found:

            if (
                best_index is None
                or row["ratings_count"]
                > books.loc[best_index, "ratings_count"]
            ):

                best_index = index


    return best_index

# ---------------------------------------------------------
# PREPARE CATALOG INFORMATION FOR RECOMMENDATIONS
# ---------------------------------------------------------

def build_catalog_info(books):

    # Count genres in the catalog
    genre_counts = {}

    for genre_list in books["genres_list"]:

        for genre in genre_list:

            genre_lower = genre.lower()

            if genre_lower in genre_counts:
                genre_counts[genre_lower] += 1
            else:
                genre_counts[genre_lower] = 1


    catalog_genres = list(genre_counts.keys())


    # Count useful words appearing in book descriptions
    common_words = {}

    for words in books["description_words"]:

        for word in words:

            if word in common_words:
                common_words[word] += 1
            else:
                common_words[word] = 1


    return catalog_genres, common_words

# Keep useful words from the user's request that
# actually occur often enough in book descriptions
def build_search_words(user_input, common_words):

    search_words = []

    for word in clean_words(user_input):

        if (
            word in common_words
            and common_words[word] >= 10
            and word not in search_words
        ):

            search_words.append(word)

    return search_words

# Turn the GUI answers into the preference structure
# used by the recommendation engine
def make_preferences(
    books,
    genre_text,
    length_text,
    minimum_rating,
    mood_text,
    loved_titles,
    selected_avoid_genres=None
):

    if selected_avoid_genres is None:
        selected_avoid_genres = []


    catalog_genres, common_words = build_catalog_info(books)


    # Understand positive and negative language
    genre_wanted, genre_unwanted = split_wanted_unwanted(
        genre_text
    )

    mood_wanted, mood_unwanted = split_wanted_unwanted(
        mood_text
    )


    # -----------------------------------------------------
    # GENRES TO AVOID
    # -----------------------------------------------------

    avoid_genres = []


    # First find genres the user rejected in natural language
    for genre in genre_synonyms:

        for synonym in genre_synonyms[genre]:

            if (
                has_phrase(genre_unwanted, synonym)
                or has_phrase(mood_unwanted, synonym)
            ):

                if genre not in avoid_genres:
                    avoid_genres.append(genre)


    # Add genres explicitly selected in the Streamlit GUI
    for genre_text_to_avoid in selected_avoid_genres:

        matched_genre = match_genre(
            genre_text_to_avoid,
            catalog_genres
        )

        if (
            matched_genre != "any"
            and matched_genre not in avoid_genres
        ):

            avoid_genres.append(matched_genre)


    # -----------------------------------------------------
    # FAVORITE BOOKS
    # -----------------------------------------------------

    loved_ids = []
    loved_names = []


    for title in loved_titles:

        index = find_book(
            books,
            title
        )

        if index is not None:

            loved_ids.append(
                books.loc[index, "book_id"]
            )

            loved_names.append(
                books.loc[index, "title"].split(" (")[0]
            )


    # -----------------------------------------------------
    # SEARCH WORDS
    # -----------------------------------------------------

    search_words = build_search_words(
        mood_wanted,
        common_words
    )


    for word in build_search_words(
        genre_wanted,
        common_words
    ):

        if word not in search_words:
            search_words.append(word)


    # -----------------------------------------------------
    # FINAL PREFERENCE DICTIONARY
    # -----------------------------------------------------

    preferences = {

        "genre": match_genre(
            genre_wanted,
            catalog_genres
        ),

        "length": match_length(
            length_text
        ),

        "minimum_rating": minimum_rating,

        "moods": match_moods(
            mood_wanted
        ),

        "tone": match_tone(
            mood_wanted
        ),

        "avoid_moods": match_moods(
            mood_unwanted
        ),

        "avoid_genres": avoid_genres,

        "loved_ids": loved_ids,

        "loved_names": loved_names,

        "search_words": search_words
    }


    return preferences

# ---------------------------------------------------------
# ASSOCIATION-RULE RECOMMENDATIONS
# ---------------------------------------------------------

def get_also_loved(preferences, rules):

    also_loved = {}


    for book_id in preferences["loved_ids"]:

        if book_id in rules:

            for other_id, confidence, lift in rules[book_id]:

                relationship_score = confidence * lift


                if other_id in also_loved:

                    also_loved[other_id] += relationship_score

                else:

                    also_loved[other_id] = relationship_score


    return also_loved

# Check whether a book has the requested genre
def book_has_genre(genre_list, wanted_genre):

    if wanted_genre == "any":
        return True


    for genre in genre_list:

        if genre.lower() == wanted_genre.lower():
            return True


    return False

# ---------------------------------------------------------
# SCORE EACH BOOK
# ---------------------------------------------------------

def score_book(row, preferences, also_loved):

    score = 0
    reasons = []


    # -----------------------------------------------------
    # Genre
    # -----------------------------------------------------

    if (
        preferences["genre"] != "any"
        and book_has_genre(
            row["genres_list"],
            preferences["genre"]
        )
    ):

        score += 5

        reasons.append(
            preferences["genre"] + " genre"
        )


    # -----------------------------------------------------
    # Reader moods
    # -----------------------------------------------------

    for mood in preferences["moods"]:

        if mood in row["moods_list"]:

            score += 5

            reasons.append(
                "readers call it " + mood
            )


    # -----------------------------------------------------
    # VADER tone
    # -----------------------------------------------------

    if (
        preferences["tone"] != "any"
        and row["tone"] == preferences["tone"]
    ):

        score += 2

        reasons.append(
            row["tone"] + " tone"
        )


    # -----------------------------------------------------
    # Readers who loved X also loved Y
    # -----------------------------------------------------

    if (
        len(preferences["loved_ids"]) > 0
        and row["book_id"] in also_loved
    ):

        score += (
            5
            * also_loved[row["book_id"]]
            / len(preferences["loved_ids"])
        )


        reasons.append(
            "fans of "
            + " & ".join(
                preferences["loved_names"][:2]
            )
            + " loved it"
        )


    # -----------------------------------------------------
    # Goodreads rating
    # -----------------------------------------------------

    score += row["average_rating"] * 0.5


    if (
        row["average_rating"]
        >= preferences["minimum_rating"]
    ):

        score += 2

    else:

        score -= 2


    # -----------------------------------------------------
    # Length preference
    # -----------------------------------------------------

    if (
        preferences["length"] == "short"
        and row["pages"] < 300
    ):

        score += 2


    elif (
        preferences["length"] == "medium"
        and row["pages"] >= 300
        and row["pages"] <= 500
    ):

        score += 2


    elif (
        preferences["length"] == "long"
        and row["pages"] > 500
    ):

        score += 2


    elif preferences["length"] == "any":

        score += 1


    # -----------------------------------------------------
    # Words/themes from the user's request
    # -----------------------------------------------------

    matched = []


    for word in preferences["search_words"]:

        if (
            word in row["description_words"]
            or word in row["keywords_list"]
        ):

            matched.append(word)


    # Maximum of 4 keyword bonus points
    score += min(
        len(matched),
        4
    )


    if len(matched) > 0:

        reasons.append(
            "mentions " + ", ".join(matched)
        )


    # -----------------------------------------------------
    # Very small popularity bonus
    # -----------------------------------------------------

    if row["ratings_count"] > 1000000:

        score += 0.5


    elif row["ratings_count"] > 100000:

        score += 0.25


    return score, reasons

# ---------------------------------------------------------
# FILTER BOOKS
# ---------------------------------------------------------

def filter_books(
    books,
    preferences,
    use_genre,
    use_length,
    use_rating
):

    keep = []


    for index, row in books.iterrows():

        ok = True


        # Never recommend sequels or books the user
        # already told us they loved
        if (
            row["is_sequel"]
            or row["book_id"] in preferences["loved_ids"]
        ):

            ok = False


        # -------------------------------------------------
        # Always respect avoided genres
        # -------------------------------------------------

        for genre in preferences["avoid_genres"]:

            if book_has_genre(
                row["genres_list"],
                genre
            ):

                ok = False


        # -------------------------------------------------
        # Always respect avoided moods
        # -------------------------------------------------

        for mood in preferences["avoid_moods"]:

            if mood in row["moods_list"]:
                ok = False


        # -------------------------------------------------
        # Preferred genre
        # -------------------------------------------------

        if (
            use_genre
            and not book_has_genre(
                row["genres_list"],
                preferences["genre"]
            )
        ):

            ok = False


        # -------------------------------------------------
        # Minimum rating
        # -------------------------------------------------

        if (
            use_rating
            and row["average_rating"]
            < preferences["minimum_rating"]
        ):

            ok = False


        # -------------------------------------------------
        # Length
        # -------------------------------------------------

        if use_length:

            if (
                preferences["length"] == "short"
                and not row["pages"] < 300
            ):

                ok = False


            elif (
                preferences["length"] == "medium"
                and not (
                    row["pages"] >= 300
                    and row["pages"] <= 500
                )
            ):

                ok = False


            elif (
                preferences["length"] == "long"
                and not row["pages"] > 500
            ):

                ok = False


        keep.append(ok)


    return books[keep]

# ---------------------------------------------------------
# GENERATE FINAL RECOMMENDATIONS
# ---------------------------------------------------------

def recommend_books(
    books,
    preferences,
    rules
):

    relaxation_notes = []


    use_genre = True
    use_length = True
    use_rating = True


    # Start with all requested conditions
    matches = filter_books(
        books,
        preferences,
        use_genre,
        use_length,
        use_rating
    )


    # -----------------------------------------------------
    # Relax length first
    # -----------------------------------------------------

    if (
        len(matches) < 5
        and preferences["length"] != "any"
    ):

        use_length = False

        relaxation_notes.append(
            "There were not enough exact matches, "
            "so the length preference was relaxed."
        )


        matches = filter_books(
            books,
            preferences,
            use_genre,
            use_length,
            use_rating
        )


    # -----------------------------------------------------
    # Then minimum rating
    # -----------------------------------------------------

    if (
        len(matches) < 5
        and preferences["minimum_rating"] > 0
    ):

        use_rating = False

        relaxation_notes.append(
            "There were still not enough matches, "
            "so the minimum rating was relaxed."
        )


        matches = filter_books(
            books,
            preferences,
            use_genre,
            use_length,
            use_rating
        )


    # -----------------------------------------------------
    # Genre is relaxed last
    # -----------------------------------------------------

    if (
        len(matches) < 5
        and preferences["genre"] != "any"
    ):

        use_genre = False

        relaxation_notes.append(
            "There were still not enough matches, "
            "so the preferred genre was relaxed."
        )


        matches = filter_books(
            books,
            preferences,
            use_genre,
            use_length,
            use_rating
        )


    # -----------------------------------------------------
    # Score qualifying books
    # -----------------------------------------------------

    books_to_score = matches.copy()


    also_loved = get_also_loved(
        preferences,
        rules
    )


    scores = []
    reasons = []


    for index, row in books_to_score.iterrows():

        score, why = score_book(
            row,
            preferences,
            also_loved
        )

        scores.append(score)

        reasons.append(
            "; ".join(why)
        )


    books_to_score["score"] = scores
    books_to_score["why"] = reasons


    books_to_score = books_to_score.sort_values(
        by=[
            "score",
            "average_rating",
            "ratings_count"
        ],
        ascending=False
    )


    # -----------------------------------------------------
    # Diversity: maximum one book per author
    # -----------------------------------------------------

    picked_rows = []
    picked_authors = []


    for index, row in books_to_score.iterrows():

        if row["author"] not in picked_authors:

            picked_rows.append(index)

            picked_authors.append(
                row["author"]
            )


        if len(picked_rows) == 5:
            break


    results = books_to_score.loc[
        picked_rows
    ]


    return results, relaxation_notes