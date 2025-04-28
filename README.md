---

# 📚 LLM BOOK RECOMMENDATION

<img width="925" alt="image" src="https://github.com/user-attachments/assets/5e737795-1d86-42e9-bb9e-75c7c94fff4c" />

The project is a book recommendation system using open-sourced LLM models from Hugging Face, LangChain, and OpenAI.  
The data is sourced from [Kaggle - 7K Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata/data), and pre-processed using Pandas in a Jupyter Notebook.

---

## 📦 Setting up the Environment

- The original tutorial uses PyCharm, but this project uses a **Conda environment** with libraries listed in `requirements.txt`.
- API keys are required:
  - Hugging Face (free)
  - OpenAI (paid, small fee)
- Save your API keys in a `.env` file for secure access.

---

## 📊 Exploratory Data Analysis (EDA)

Several features in the dataset had missing values. Here's a summary of the actions taken:

- **Subtitle**: Many missing; combined into the main title in a new column.
- **Description**: Rows with missing descriptions were removed (~<5% of the data).
- **Categories**: Over 567 unique entries; simplified using text classification later.
- **Average Rating, Number of Pages, Ratings Count**: Rows with missing values were removed.
- **Thumbnail**: Missing thumbnails are replaced with a placeholder image in the Gradio app.

---

## 🔥 Main Components

The book recommendation system consists of three main components:

---

### 1. Vector Search

A **semantic vector search pipeline** was built to recommend books based on their content.

- **Data Preparation**:
  - Load `books_cleaned.csv`
  - Save `tagged_description` into a plain text file.

- **Text Processing**:
  - Use `TextLoader` and `CharacterTextSplitter` to treat each description independently.

- **Vector Embeddings**:
  - Text chunks are converted into dense vectors (e.g., using `OpenAIEmbeddings`).

- **Vector Database**:
  - Store vectors in **ChromaDB** for fast retrieval.

- **Semantic Search Function**:
  - `retrieve_semantic_recommendations(query, top_k)` searches for the top 50 matches.
  - Results are mapped back to the original book dataset.

✅ **Use Case**: Content-based recommendations using NLP embeddings.

---

### 2. Text Classification

Simplifies and corrects book categories using **zero-shot learning**.

- **Category Mapping**:
  - Simplified into broader categories like `Fiction`, `Nonfiction`, etc.

- **Zero-Shot Classification**:
  - Using `facebook/bart-large-mnli` from Hugging Face.
  - No labeled training data needed.
  - Assigns labels based on model's probability scores.

- **Workflow**:
  - Predict missing or ambiguous categories.
  - Achieved an accuracy of **77.83%** on test examples.

✅ **Use Case**: Clean and enrich metadata for better recommendations.

---

### 3. Sentiment Analysis

Analyzes **emotion** in book descriptions using a fine-tuned model.

- **Model Used**:
  - `j-hartmann/emotion-english-distilroberta-base` (Hugging Face).

- **Emotion Labels**:
  - `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, `neutral`.

- **Processing**:
  - Split descriptions into sentences.
  - Predict emotions for each sentence.
  - Aggregate scores to find dominant emotions.

- **Data Merging**:
  - Emotion scores are merged back into the main dataset.

✅ **Use Case**: Enable emotion/mood-based book search and discovery.

---

## 🎛 Web App Interface

- The final book recommender is deployed locally using **Gradio**.
- Allows easy search and interaction through a simple web interface.

---

# ✨ Summary

This project demonstrates how **modern LLM tools** and **open-source models** can be combined for:

- Intelligent content-based book recommendations
- Automated text classification
- Emotion-driven sentiment analysis
- Building a practical, interactive web app with minimal effort

---






