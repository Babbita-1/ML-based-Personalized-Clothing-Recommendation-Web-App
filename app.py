import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# -------------------------
# Recommender Classss
# -------------------------
class EcommerceRecommendationSystem:
    def __init__(self):
        self.embedding_files = {
            "Product Features": "ecommerce_embeddings.npy",
        }
        self.metadata = pd.read_csv("ecommerce_data.csv")
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = np.load(self.embedding_files["Product Features"])

    def vector_search(self, query, top_n=9, similarity_threshold=0.4):
        if not query.strip():
            return []

        query_embedding = self.sentence_model.encode(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        valid_indices = np.where(similarities >= similarity_threshold)[0]
        if len(valid_indices) == 0:
            return []

        sorted_indices = valid_indices[similarities[valid_indices].argsort()[::-1]]
        top_indices = sorted_indices[:top_n]
        results = self.metadata.iloc[top_indices]
        return results


# -------------------------
# Streamlit App UI
# -------------------------
st.set_page_config(page_title="Outfit Recommender", layout="wide")
st.image("logo.png")




# -------------------------
# Intro Section
# -------------------------
st.markdown("""
### 👋 Welcome to the RecoWear!
This application helps you easily find the best fashion products based on your descriptions. 
Whether you're looking for **casual wear, formal shirts, ethnic kurtas, or sporty outfits**, 
just describe it — and let the AI recommend the perfect products for you.

This project aims to simplify fashion discovery using Artificial Intelligence and Machine Learning 
by transforming text descriptions into smart outfit recommendations.
""")

# -------------------------
# "How it was developed" Button
# -------------------------
with st.expander("🛠️ How this product was developed"):
    st.markdown("""
    -  **Technology Used:**  
      - Sentence Transformers (`all-MiniLM-L6-v2`) for converting product descriptions into vector embeddings.
      - Scikit-learn for similarity search using **cosine similarity**.
      - Built completely with **Python** and **Streamlit** for the interactive web interface.
    
    -  **Data:**  
      - Product metadata like product name, category, color, season, usage, etc.
      - Generated embeddings for product descriptions.

    -  **Goal:**  
      To provide users an intuitive way to search for products based on natural language descriptions without needing to use filters or dropdowns.
    
    -  **Crafted with care** to help users enjoy hassle-free fashion exploration.
    """)

# -------------------------
# Load recommender
# -------------------------
recommender = EcommerceRecommendationSystem()

# -------------------------
# Sidebar - Control Panel
# -------------------------

st.sidebar.header(" Search Settings")

with st.sidebar:
    st.markdown("###  **Select Number of Products**")
    top_n_option = st.slider(
        "Number of Products",
        min_value=3,
        max_value=15,
        value=9
    )

    st.markdown("---")
    st.markdown("###  Similarity Threshold")
    st.markdown(" How similar do you want the product to be?")
    similarity_threshold = st.slider(
        "Filter by similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05
    )


    st.markdown("---")
    st.markdown("###  **Try this Searches:**")
    if st.button("👔 Formal Shirts"):
        example_query = "formal shirts navy"
    elif st.button("👗 Women Kurtas"):
        example_query = "ethnic kurta green"
    elif st.button("🎽 Sports T-shirt"):
        example_query = "sports t-shirt"
    else:
        example_query = ""

    

    st.markdown("---")
    st.markdown(
        "<br><center>Designed & Developed with ❤️<br>by **Babita**</center>",
        unsafe_allow_html=True
    )

# -------------------------
# Main Search
# -------------------------
st.subheader(" Describe What You're Looking For")
query = st.text_input(
    "Example: 'Blue denim jeans', 'Running shoes', 'Ethnic women kurta'",
    value=example_query
)

# -------------------------
# Show Results
# -------------------------
if query:
    results = recommender.vector_search(
        query=query,
        top_n=top_n_option,
        similarity_threshold=similarity_threshold
    )

    if len(results) == 0:
        st.warning("❌ No matching products found.")
    else:
        st.subheader("🎉 Recommended Products")
        cols = st.columns(3)

        for idx, (_, row) in enumerate(results.iterrows()):
            with cols[idx % 3]:
                st.image(row.get('link', ''), width=250, caption=row.get('productDisplayName', 'No Name'))
                st.markdown(f"""
                **{row.get('productDisplayName', 'No Name')}**  
                - **Category:** {row.get('masterCategory', '')} / {row.get('subCategory', '')}  
                - **Type:** {row.get('articleType', '')}  
                - **Color:** {row.get('baseColour', '')}  
                - **Season:** {row.get('season', '')} | **Year:** {row.get('year', '')}  
                - **Usage:** {row.get('usage', '')}  
                """)
else:
    st.info("💡 Enter a product description or choose from examples in the sidebar.")



