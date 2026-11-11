
# 📚 Book Finder — Personalized Book Recommendation System

An interactive, data-driven book recommendation application built with **Python, Streamlit, Natural Language Processing (NLP), and Association-Rule Mining**.

Book Finder helps readers discover books aligned with their interests, preferences, and reading mood through an intuitive interface.

## 🚀 Live Application

**[Explore Book Finder](https://ou-book-finder.streamlit.app/)**

## 🎯 Project Overview

Choosing a book can be overwhelming when readers have thousands of options and limited information about which titles match their preferences.

Book Finder addresses this challenge by combining multiple recommendation techniques to create a more personalized book discovery experience.

The application considers user preferences, book characteristics, text-based signals, and relationships between books to recommend relevant titles.

### Project Objectives

- Help readers discover books matching their interests.
- Provide personalized suggestions using multiple data-driven techniques.
- Apply NLP to analyze descriptive text and reader preferences.
- Explore association-rule mining for recommendation generation.
- Create an accessible application through Streamlit.

## ✨ Key Features

### 1. Personalized Book Recommendations

Users can specify reading preferences to help the application identify relevant books.

### 2. Mood-Based Discovery

The recommendation experience incorporates reading moods and text analysis to help users discover books suited to their interests.

### 3. Favorite Book-Based Discovery

Users can explore recommendations informed by books they already enjoy.

### 4. Association-Rule Mining

The application uses the Apriori algorithm and association rules to explore relationships between books and reader preferences.

### 5. Text Analytics

NLTK-based text processing and sentiment-related analysis contribute additional signals for recommendations.

### 6. Interactive Streamlit Interface

The application provides a user-friendly interface for selecting preferences and exploring recommended titles.

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Web Application | Streamlit |
| Data Processing | Pandas |
| Natural Language Processing | NLTK |
| Association-Rule Mining | mlxtend |
| Recommendation Logic | Preference-Based Scoring & Association Rules |
| Version Control | Git & GitHub |

## 🧠 How the Recommendation System Works

The application follows a data-processing and recommendation workflow.

### Step 1: Data Loading

Book-related datasets are loaded and prepared for analysis.

### Step 2: Data Preprocessing

The application cleans and transforms the book data, including handling missing values and processing relevant book attributes.

### Step 3: Text Analysis

NLP techniques help extract useful signals from descriptive text and reader preferences.

### Step 4: Association-Rule Analysis

The Apriori algorithm identifies frequent itemsets and association rules that can support book recommendations.

### Step 5: User Preference Processing

The application interprets the user's selected preferences and reading interests.

### Step 6: Recommendation Generation

Books are filtered, evaluated, and ranked using the available recommendation signals.

### Step 7: Interactive Presentation

The Streamlit interface displays a personalized selection of recommended books.

## 🏗️ Project Structure

```text
book-finder/
│
├── app.py
│   └── Streamlit user interface
│
├── book_finder_engine.py
│   └── Data processing and recommendation logic
│
├── requirements.txt
│   └── Python dependencies
│
└── .streamlit/
    └── Streamlit configuration
```

## 💻 Running the Application Locally

### Prerequisites

- Python 3
- pip
- Internet access for retrieving required datasets and NLTK resources

### 1. Clone the Repository

```bash
git clone https://github.com/Vaibhavvdighe/book-finder.git
```

### 2. Navigate to the Project Directory

```bash
cd book-finder
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Application

```bash
streamlit run app.py
```

Open the local Streamlit URL displayed in your terminal.

**Note:** The application retrieves external book datasets. Availability of the upstream datasets is required for successful execution.

## 📊 Data & Recommendation Techniques

This project explores multiple analytical approaches, including:

- Data cleaning and preprocessing
- Natural language processing
- Text and sentiment analysis
- Frequent itemset mining
- Association-rule generation
- Preference matching
- Recommendation scoring and ranking

Rather than relying on a large language model, this application demonstrates how traditional NLP, analytical methods, and recommendation logic can support personalized discovery.

## 📈 Skills Demonstrated

**Data Analytics**
- Preparing and transforming datasets
- Extracting relevant information from structured and textual data

**Machine Learning & Recommendation Methods**
- Applying association-rule mining
- Developing and combining recommendation signals

**Software Development**
- Structuring Python application logic
- Building an interactive application with Streamlit

**Product Thinking**
- Identifying a user discovery problem
- Translating user preferences into application functionality
- Designing an accessible recommendation experience

## 🔮 Potential Future Improvements

- Additional recommendation algorithms
- More advanced semantic similarity techniques
- Improved recommendation evaluation
- User profiles and saved reading lists
- Expanded book datasets
- Further performance optimization

## 👨‍💻 Author

**Vaibhav Dighe**

MBA + M.S. in Management of Information and Technology  
University of Oklahoma

[LinkedIn](https://www.linkedin.com/in/vaibhavdighe/) | [GitHub](https://github.com/Vaibhavvdighe)

---

*Helping readers discover their next great book through data and technology.*
