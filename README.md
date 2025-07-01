# 👗 ML-based Personalized Clothing Recommendation Web App

Welcome to the **ML-Based Personalized Clothing Recommendation System**, a smart outfit discovery platform powered by **Machine Learning** and **Natural Language Processing**. This web application recommends fashion products based on natural language queries — making fashion search effortless and intuitive!

---

## Demo

Try it live on Streamlit: [Streamlit App 🔗](https://ml-cloth-recommendation.streamlit.app/)  


---

## Features

- **Natural Language Search**: Just type in what you're looking for (e.g., "green kurta for summer") and get personalized outfit suggestions.
- **Semantic Similarity Matching**: Uses state-of-the-art Sentence Transformers and Cosine Similarity for relevant product matches.
- **Adjustable Controls**: 
  - Choose number of product results
  - Customize similarity threshold
- **Interactive UI**: Built with **Streamlit** for a clean, responsive user interface.
- **Optional ML Model**: A simple classifier to predict whether a product is "Intimate Wear".

---

## Project Structure

<pre>
ML-based-Personalized-Clothing-Recommendation-Web-App/
│
├── ecommerce_data.csv              # Product metadata
├── ecommerce_embeddings.npy        # Precomputed sentence embeddings
├── app.py                          # Main Streamlit app file
├── generate_embeddings.py          # Script to generate sentence embeddings using HuggingFace
├── requirements.txt                # Required Python libraries
└── README.md                       # Project documentation
</pre>

---

##  Tech Stack

- **Frontend**: Streamlit
- **ML/NLP**: 
  - Sentence Transformers (`all-MiniLM-L6-v2`)
  - Scikit-learn
  - HuggingFace Transformers
- **Backend**: Python
- **Libraries**: Pandas, NumPy, Torch

---

## Installation

###  Prerequisites

- Python 3.8+
- pip or conda

###  Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Babbita-1/ML-based-Personalized-Clothing-Recommendation-Web-App.git
   cd ML-based-Personalized-Clothing-Recommendation-Web-App
   ```
2. Create a virtual environment
 ```python -m venv venv
      source venv/bin/activate        # On Windows: venv\Scripts\activate
  ```

3. Create a virtual environment
```pip install -r requirements.txt ```

4. Run the Streamlit app
```streamlit run app.py```



## Example Queries

- `blue denim jeans`
- `green ethnic kurta`
- `white sports t-shirt`
- `summer floral dress`
- `formal navy shirts for men`

---

## Contributing

Contributions are welcome!  
Feel free to fork the repo, open an issue, or submit a pull request to improve this project.

---

##  License

This project is open-source and available under the [MIT License](LICENSE).

