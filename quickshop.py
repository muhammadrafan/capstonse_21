import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load IndoBERT Model
MODEL_NAME = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Daftar kata positif & negatif untuk membantu sentimen analisis
positive_words = ["mantap", "bagus", "jernih", "nyaman", "original", "premium", "cepat", "murah", "berkualitas"]
negative_words = ["buruk", "jelek", "rusak", "lemot", "cacat", "lambat", "kurang", "tidak", "gagal"]

# Function to replace emojis with words
def replace_emojis(text):
    emoji_dict = {
        "👍": "bagus", "😍": "sangat suka", "😋": "puas", "😊": "senang", "💯": "mantap", "🔥": "keren",
        "❌": "buruk", "😡": "marah", "😭": "sedih", "👎": "jelek"
    }
    for emoji, replacement in emoji_dict.items():
        text = text.replace(emoji, f" {replacement} ")
    return text

# Function to preprocess text
def preprocess_text(text):
    text = replace_emojis(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to scrape Tokopedia reviews
def scrape_tokopedia_reviews(product_url, max_reviews=100):
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.get(product_url)
    time.sleep(5)

    reviews_data = []
    while len(reviews_data) < max_reviews:
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(3)
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        containers = soup.find_all("article", class_="css-72zbc4")
        
        if not containers:
            break  # Stop if no more reviews are found
        
        for container in containers:
            if len(reviews_data) >= max_reviews:
                break
            try:
                name_elem = container.select_one("div.css-u2c4jt span.name")
                name = name_elem.text.strip() if name_elem else "Unknown"
                
                rating_elem = container.select_one("div[data-testid='icnStarRating']")
                rating = rating_elem["aria-label"] if rating_elem else "Tidak Ada Rating"
                rating = int(re.search(r'\d+', rating).group()) if rating != "Tidak Ada Rating" else 0
                
                review_elem = container.select_one("p span[data-testid='lblItemUlasan']")
                review_text = review_elem.text.strip() if review_elem else "No Review"
                
                reviews_data.append({"Nama": name, "Rating": rating, "Ulasan": review_text})
            except Exception as e:
                print("Error extracting review:", e)
                continue
    
    driver.quit()
    return reviews_data

# Function to analyze sentiment
def analyze_sentiment(reviews):
    sentiments = []
    preprocessed_texts = []
    positive_counts = []
    negative_counts = []
    sentiment_labels = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
    
    for review in reviews:
        clean_text = preprocess_text(review["Ulasan"])
        preprocessed_texts.append(clean_text)
        
        positive_count = sum(1 for word in positive_words if word in clean_text)
        negative_count = sum(1 for word in negative_words if word in clean_text)
        positive_counts.append(positive_count)
        negative_counts.append(negative_count)
        
        inputs = tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        sentiment = torch.argmax(logits).item()
        confidence = torch.softmax(logits, dim=-1)
        max_confidence = confidence[sentiment].item()

       # Koreksi prediksi sentimen berdasarkan confidence, rating, dan jumlah kata positif/negatif
        rating = review["Rating"]
        if max_confidence < 0.6:  # Jika confidence rendah, buat netral
            sentiment = 1
        if sentiment == 0 and (rating == 3):
            sentiment = 1  # Ubah negatif jadi netral jika rating 3 atau lebih banyak kata positif
        elif sentiment == 2 and (rating == 3 ):
            sentiment = 1  # Ubah positif jadi netral jika rating 3 atau lebih banyak kata negatif
        elif sentiment == 0 and rating in [4, 5] or positive_count >= 2:
            sentiment = 1  # Ubah negatif jadi netral jika rating 4 atau 5 dengan cukup kata positif
        elif sentiment == 2 and rating in [1, 2] or negative_count >= 2:
            sentiment = 1  # Ubah positif jadi netral jika rating 1 atau 2 dengan cukup kata negatif
        elif sentiment == 1 and rating in [4, 5] or positive_count >= 3:
            sentiment = 2  # Ubah netral jadi positif jika rating 4 atau 5 dengan banyak kata positif
        elif sentiment == 1 and rating in [1, 2] or negative_count >= 3:
            sentiment = 0  # Ubah netral jadi negatif jika rating 1 atau 2 dengan banyak kata negatif

        sentiments.append(sentiment_labels[sentiment])
    return sentiments, preprocessed_texts, positive_counts, negative_counts
      
# Function to generate word cloud
def generate_wordcloud(texts):
    text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Streamlit UI
st.title("🛍️ QuickShop: Evaluasi Produk Tokopedia dengan AI")
st.write("Masukkan link produk Tokopedia untuk melihat ulasan dan analisis sentimen.")

product_url = st.text_input("🔗 Masukkan link produk Tokopedia:")

if st.button("🚀 Analisis Sentimen"):
    if product_url:
        with st.spinner("Mengambil ulasan..."):
            reviews = scrape_tokopedia_reviews(product_url, max_reviews=100)

        st.write(f"📊 Total ulasan ditemukan: {len(reviews)}")
        
        if len(reviews) == 0:
            st.warning("⚠️ Tidak ada ulasan yang ditemukan. Coba gunakan link produk lain atau pastikan halaman memiliki ulasan.")
        else:
            with st.spinner("🤖 Menganalisis sentimen..."):
                sentiment_results, preprocessed_texts, positive_counts, negative_counts = analyze_sentiment(reviews)

            for i in range(len(reviews)):
                reviews[i]["Sentimen"] = sentiment_results[i]
                reviews[i]['Potivie Count'] = positive_counts[i]
                reviews[i]['Negative Count'] = negative_counts[i]
                reviews[i]["Ulasan Setelah Preprocessing"] = preprocessed_texts[i]

            sentiment_df = pd.DataFrame(reviews)
            st.dataframe(sentiment_df)

            summary = f"""
            ✅ **{sentiment_results.count('Positif')}** ulasan positif  
            🟡 **{sentiment_results.count('Netral')}** ulasan netral  
            ❌ **{sentiment_results.count('Negatif')}** ulasan negatif  
            """
            st.subheader("📌 Ringkasan Sentimen")
            st.markdown(summary)

            # Generate and display word cloud
            st.subheader("📌 Word Cloud dari Ulasan")
            generate_wordcloud(preprocessed_texts)

            sentiment_df.to_csv("Tokopedia_Reviews.csv", index=False)
            st.success("✅ Analisis selesai! Data ulasan disimpan sebagai 'Tokopedia_Reviews.csv'.")
    else:
        st.error("⚠️ Harap masukkan link produk Tokopedia yang valid!")
