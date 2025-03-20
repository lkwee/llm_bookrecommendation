### LLM BOOK RECOMMENDATION

The project is a book recommendation using open sourced LLM models from HuggingFace, Langchain, OpenAI, by following a tutorial video by freeCodeCamp.org.
The data is obtained from Kaggle, which is then examined, cleaned and pre-processed using Pandas library in Jupyter Notebook.
<br> 
There are three components in this project: vector search for semantic book recoomendation, text classification to refine and simplify book categories and sentiment analysis to classify the tone of each book.
<br> 
1. **Vector Search**
   <br> The script implements a **vector search pipeline** for retrieving books based on **semantic similarity**. It follows these key steps:  
     **Data Preparation**: The script loads a cleaned book dataset (`books_cleaned.csv`), extracts the `tagged_description` column, and saves it as a plain text file (`tagged_description.txt`).  
     **Text Processing**: The `TextLoader` reads the text file, and the `CharacterTextSplitter` splits the descriptions into chunks based on newline separators, ensuring that each description is treated as an independent document.  
     **Vector Embeddings**: Though not explicitly shown, an embedding model (e.g., `OpenAIEmbeddings`) would convert these text chunks into dense numerical vectors.  
     **Vector Database**: The embeddings are stored in **Chroma**, a vector database designed for fast similarity searches.  
     **Semantic Search Function (`retrieve_semantic_recommendations`)**:  
     - Given a `query`, it searches the vector database (`db_books.similarity_search`) to find the **top 50** most relevant book descriptions.  
     - The retrieved results contain text data, so it extracts the first word (assumed to be an ISBN or book ID).  
     - The function returns a DataFrame of books whose ISBNs match the extracted IDs, providing the **top `top_k` recommendations** based on semantic similarity.  
  This approach enables **content-based recommendations** using NLP embeddings, making it useful for applications like **book discovery, personalized recommendations, or search engines**.

2. **Text Classification**
  <br> The original book categories are numerous (567 unique categories) and the aim here is to simplify them in order to complement the book recommendation system.
  This script implements **text classification** for books using a **zero-shot learning approach** with `facebook/bart-large-mnli`, a pre-trained NLP model from Hugging Face.
     **Category Mapping**: The `category_mapping` dictionary simplifies book genres into broader categories (`Fiction`, `Nonfiction`, `Children's Fiction`, etc.), which are assigned to books via `.map()`.
     These broarder mapping is based on the top categories in the original dataset.
     **Zero-Shot Classification Pipeline**: The `facebook/bart-large-mnli` model is loaded using Hugging Face’s `pipeline`, with `device="mps"` to run on Apple Silicon GPUs. 
     **Single Prediction Example**:  
     - A sample book description from the `"Fiction"` category is selected.  
     - The model predicts whether it falls under `"Fiction"` or `"Nonfiction"` by returning probability scores for each category.  
     - The category with the highest score (`np.argmax()`) is assigned as the predicted label.  
     **Classification (Fiction & Nonfiction Books)**:  
     - The script loops through book from `"Fiction"` and `"Nonfiction"` categories.  
     - Each description is classified using `generate_predictions()`, and results are stored in a DataFrame.
     - The predicted and actual labels are then compared. The accuracy is 77.83%, which is quite good.
     **Using the prediction to complete the missing categories**:  
     - Books without a `"simple_categories"` label are identified.  
     - The model predicts categories for these books.  
     - The missing categories are merged back into the original dataset.  
  This approach allows **automatic categorization** of books **without labeled training data** using **zero-shot learning**, making it ideal for **unstructured datasets** where categories are missing or inconsistent. 
     
3. **Sentiment Analysis**
  <br> This script performs **sentiment analysis** on book descriptions using a **pre-trained emotion classification model** (`j-hartmann/emotion-english-distilroberta-base`) from Hugging Face.  
     **Load Dataset**: Reads `books_with_categories.csv`, which contains book descriptions and ISBNs.  
     **Initialize Sentiment Classifier**:  
     - Uses a **DistilRoBERTa-based** model fine-tuned for **emotion detection**.  
     - Classifies text into **seven emotion labels**: `"anger"`, `"disgust"`, `"fear"`, `"joy"`, `"sadness"`, `"surprise"`, and `"neutral"`.  .  
     **Processing Descriptions**:  
     - Each book's description is **split into sentences**.  
     - The classifier predicts emotions for each sentence.  
     - The function `calculate_max_emotion_scores()` extracts the **highest** predicted score per emotion across all sentences.  
     **Batch Processing**:  
     - Loops through **all books**, predicting **emotion scores** for each description.  
     - Stores results in a dictionary with **ISBNs as identifiers**.  
     - Uses `tqdm` for a **progress bar** when processing large datasets.  
     **DataFrame Creation & Merging**:  
     - Converts **emotion scores** into a DataFrame.  
     - Merges results with the original book dataset on `isbn13`, ensuring each book has its associated **emotion scores**.  
  This method enables **emotion-based book categorization**, useful for **reader sentiment analysis**, **recommendation systems**, or **mood-based book searches**. 

Lastly, Gradio is used to run the web interface locally to access the final book recommendation.
