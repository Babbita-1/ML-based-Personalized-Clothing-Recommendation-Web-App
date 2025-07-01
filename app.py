import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# -------------------------
# Recommender Class
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
st.title("👗🧥👚 **Outfit Recommender**")

# -------------------------
# Intro Section
# -------------------------
st.markdown("""
### 👋 Welcome to the Outfit Recommender!
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
    - 🧠 **Technology Used:**  
      - Sentence Transformers (`all-MiniLM-L6-v2`) for converting product descriptions into vector embeddings.
      - Scikit-learn for similarity search using **cosine similarity**.
      - Built completely with **Python** and **Streamlit** for the interactive web interface.
    
    - 📦 **Data:**  
      - Product metadata like product name, category, color, season, usage, etc.
      - Generated embeddings for product descriptions.

    - 🎯 **Goal:**  
      To provide users an intuitive way to search for products based on natural language descriptions without needing to use filters or dropdowns.
    
    - ❤️ **Crafted with care** to help users enjoy hassle-free fashion exploration.
    """)

# -------------------------
# Load recommender
# -------------------------
recommender = EcommerceRecommendationSystem()

# -------------------------
# Sidebar - Control Panel
# -------------------------
st.sidebar.header("🔧 Search Settings")

with st.sidebar:
    st.markdown("### 🔍 **Select Number of Products**")
    top_n_option = st.slider(
        "Number of Products",
        min_value=3,
        max_value=15,
        value=9
    )

    st.markdown("---")
    st.markdown("### 🧠 Similarity Threshold")
    st.markdown(" How similar do you want the product to be?")
    similarity_threshold = st.slider(
        "Filter by similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05
    )


    st.markdown("---")
    st.markdown("### 🎯 **Try this Searches:**")
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
st.subheader("📝 Describe What You're Looking For")
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




# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sentence_transformers import SentenceTransformer


# # -------------------------
# # Recommender System Class
# # -------------------------
# class EcommerceRecommendationSystem:
#     def __init__(self):
#         self.embedding_files = {
#             "Product Features": "ecommerce_embeddings.npy",
#         }

#         self.metadata = pd.read_csv("ecommerce_data.csv")
#         self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
#         self.embeddings = np.load(self.embedding_files["Product Features"])

#     def vector_search(self, query, top_n=5, similarity_threshold=0.4):
#         if not query.strip():
#             return []

#         query_embedding = self.sentence_model.encode(query)
#         similarities = cosine_similarity([query_embedding], self.embeddings)[0]

#         valid_indices = np.where(similarities >= similarity_threshold)[0]
#         if len(valid_indices) == 0:
#             return []

#         sorted_indices = valid_indices[similarities[valid_indices].argsort()[::-1]]
#         top_indices = sorted_indices[:top_n]
#         results = self.metadata.iloc[top_indices]
#         return results


# # -------------------------
# # ML Model Class
# # -------------------------
# class EcommerceMLModel:
#     def __init__(self, data):
#         self.data = data
#         self.model = None

#     def prepare_data(self):
#         df = self.data.copy()
#         df = df.select_dtypes(include=[np.number])
#         df = df.dropna()

#         if 'is_intimate_wear' not in self.data.columns:
#             st.error("'is_intimate_wear' column is missing in the data.")
#             return None, None, None, None

#         X = df.drop(columns=['is_intimate_wear'], errors='ignore')
#         y = self.data['is_intimate_wear'].astype(int)

#         return train_test_split(X, y, test_size=0.2, random_state=42)

#     def train_model(self):
#         X_train, X_test, y_train, y_test = self.prepare_data()

#         if X_train is None:
#             return None

#         self.model = RandomForestClassifier()
#         self.model.fit(X_train, y_train)

#         y_pred = self.model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)

#         return accuracy

#     def predict(self, input_data):
#         if self.model is None:
#             return None
#         return self.model.predict([input_data])[0]


# # -------------------------
# # Streamlit App
# # -------------------------
# st.set_page_config(page_title="E-commerce App", layout="wide")
# st.title("🛍️ E-commerce Recommender & ML App")

# # Initialize classes
# recommender = EcommerceRecommendationSystem()
# ml_model = EcommerceMLModel(recommender.metadata)

# # Sidebar
# st.sidebar.header("Options")
# app_mode = st.sidebar.selectbox("Choose Mode", ["Recommender", "ML Model"])

# # -------------------------
# # Recommender Mode
# # -------------------------
# if app_mode == "Recommender":
#     st.subheader("🔍 Product Recommendation")

#     query = st.text_input("Describe what you're looking for:")
#     top_n = st.slider("Number of Results", 1, 20, 5)
#     similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.4, 0.05)

#     if query:
#         results = recommender.vector_search(
#             query=query,
#             top_n=top_n,
#             similarity_threshold=similarity_threshold
#         )

#         if len(results) == 0:
#             st.warning("No matching products found.")
#         else:
#             st.subheader("Recommended Products")
#             for _, row in results.iterrows():
#                 product_name = row.get('productDisplayName', 'Name Not Found')
#                 category = row.get('masterCategory', 'N/A') + " - " + row.get('subCategory', 'N/A')
#                 article = row.get('articleType', 'N/A')
#                 color = row.get('baseColour', 'N/A')
#                 season = row.get('season', 'N/A')
#                 year = row.get('year', 'N/A')
#                 usage = row.get('usage', 'N/A')
#                 image_link = row.get('link', '')

#                 st.markdown(f"""
#                 ### 🛍️ **{product_name}**  
#                 - Category: *{category}*  
#                 - Article Type: *{article}*  
#                 - Color: {color}  
#                 - Season: {season} | Year: {year}  
#                 - Usage: {usage}  
#                 """)

#                 if pd.notnull(image_link) and image_link != "":
#                     st.image(image_link, width=250)
#     else:
#         st.info("Enter a product description above to get recommendations.")

# # -------------------------
# # ML Model Mode
# # -------------------------
# elif app_mode == "ML Model":
#     st.subheader("🤖 Predict 'Is Intimate Wear?'")

#     if st.button("Train Model"):
#         accuracy = ml_model.train_model()
#         if accuracy:
#             st.success(f"Model trained with accuracy: {accuracy:.2f}")
#         else:
#             st.error("Model training failed due to missing data.")

#     st.write("### Input Features for Prediction")

#     numeric_columns = recommender.metadata.select_dtypes(include=[np.number]).drop(columns=['is_intimate_wear'], errors='ignore').columns.tolist()

#     if numeric_columns:
#         input_data = []
#         for col in numeric_columns:
#             val = st.number_input(f"Input {col}:", value=0.0)
#             input_data.append(val)

#         if st.button("Predict"):
#             if ml_model.model:
#                 prediction = ml_model.predict(input_data)
#                 label = "Intimate Wear" if prediction == 1 else "Not Intimate Wear"
#                 st.success(f"Prediction: {label}")
#             else:
#                 st.warning("Please train the model first.")
#     else:
#         st.warning("No numeric columns available in the dataset for training.")

