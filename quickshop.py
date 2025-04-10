import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForCausalLM
import numpy as np
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from huggingface_hub import login

login(token="//masukin token")

# Load IndoBERT Model
# MODEL_NAME = "indobenchmark/indobert-base-p1"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Load fine-tuned IndoBERT Model
MODEL_DIR = "quickshop-indobert-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)



# Load model directly
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer_llm = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model_llm = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
tokenizer_llm.pad_token = tokenizer_llm.eos_token


# Daftar kata positif & negatif untuk membantu sentimen analisis
positive_words = ["mantap", "bagus", "jernih", "nyaman", "original", "premium", "cepat", "murah", "berkualitas","aman","good","good quality","terimakasih","terima kasih","memuaskan"]
negative_words = ["buruk", "jelek", "rusak", "lemot", "cacat", "lambat", "kurang", "tidak bagus", "gagal"]

# Function to replace emojis with words
def replace_emojis(text):
    emoji_dict = {
    "😍": "senang", "❤️": "cinta", "💔": "sedih", "😡": "marah", "😢": "sedih",
    "😊": "senang", "😁": "senang", "😭": "sedih", "👍": "bagus", "👎": "jelek",
    "🥰": "senang", "💖": "cinta", "💗": "cinta", "💕": "cinta", "💞": "cinta", "😞": "kecewa"
    }
    for emoji, replacement in emoji_dict.items():
        text = text.replace(emoji, f" {replacement} ")
    return text

# Function to preprocess text
def preprocess_text(text):
    text = str(text).lower()
    text = replace_emojis(text)
    text = text.replace("gk", "tidak").replace("tdk", "tidak")
    text = text.replace("bgt", "banget").replace("bgtt", "banget").replace("bgttt", "banget")
    text = text.replace("ok", "oke")
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to scrape Tokopedia reviews
def scrape_tokopedia_reviews(product_url, max_reviews=100):
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.get(product_url)
    time.sleep(7)

    driver.execute_script("window.scrollBy(0, 2000);")
    time.sleep(5)  # Tunggu beberapa detik agar konten dimuat

    # Tekan tombol "Lihat Selengkapnya" untuk deskripsi produk
    try:
        see_more_button = driver.find_element(By.XPATH, "//button[@data-testid='btnPDPSeeMore']")
        see_more_button.click()
        time.sleep(5)  # Tunggu beberapa detik agar deskripsi lengkap dimuat
    except Exception as e:
        print("Error menekan tombol 'Lihat Selengkapnya':", e)
    
    # Ambil deskripsi produk
    soup = BeautifulSoup(driver.page_source, "html.parser")
    description_elem = soup.select_one("div[data-testid='lblPDPDescriptionProduk']")
    description = description_elem.get_text(strip=True) if description_elem else "Deskripsi tidak ditemukan"

    reviews_data = []
    collected_reviews = set()
    reviews_per_page = 10  # Asumsi ada 10 ulasan per halaman

    # Ambil jumlah total ulasan dari elemen "Menampilkan X dari Y ulasan"
    try:
        total_elem = soup.find("p", {"data-testid": "reviewSortingSubtitle"})
        if total_elem:
            total_text = total_elem.get_text()
            match = re.search(r'dari (\d+)', total_text)
            if match:
                total_available = int(match.group(1))
                max_reviews = min(max_reviews, total_available)  # Pastikan tidak melebihi jumlah ulasan yang ada
    except Exception as e:
        print("Gagal membaca total ulasan:", e)

    # Hitung berapa banyak halaman yang perlu diklik berdasarkan jumlah ulasan dan ulasan per halaman
    total_pages = (max_reviews // reviews_per_page) + (1 if max_reviews % reviews_per_page > 0 else 0)
    for page in range(total_pages):
        soup = BeautifulSoup(driver.page_source, "html.parser")
        containers = soup.find_all("article", class_="css-15m2bcr")

        if not containers:
            break

        for container in containers:
            if len(reviews_data) >= max_reviews:  # Jika sudah cukup ulasan, berhenti
                break
            try:
                review_elem = container.select_one("p span[data-testid='lblItemUlasan']")
                review_text = review_elem.text.strip() if review_elem else "No Review"

                if review_text in collected_reviews:
                    continue  # Hindari ulasan duplikat

                name_elem = container.select_one("div.css-k4rf3m span.name")
                name = name_elem.text.strip() if name_elem else "Unknown"

                rating_elem = container.select_one("div[data-testid='icnStarRating']")
                rating = rating_elem["aria-label"] if rating_elem else "Tidak Ada Rating"
                rating = int(re.search(r'\d+', rating).group()) if rating != "Tidak Ada Rating" else 0

                reviews_data.append({"Nama": name, "Rating": rating, "Ulasan": review_text})
                collected_reviews.add(review_text)
            except Exception as e:
                print("Error extracting review:", e)
                continue

        if len(reviews_data) < max_reviews:  # Jika belum mencapai jumlah ulasan maksimal
            # Klik tombol "Laman berikutnya" untuk pindah ke halaman berikutnya
            try:
                next_page_button = driver.find_element(By.XPATH, "//button[@aria-label='Laman berikutnya']")
                next_page_button.click()
                time.sleep(5)  # Tunggu beberapa detik untuk halaman berikutnya dimuat
            except Exception as e:
                print("Error mengklik halaman berikutnya:", e)
                break

    driver.quit()
    return reviews_data,description

import torch

def generate_conclusion_with_llm(description, sentiment_summary):
    # Buat prompt teks natural
    prompt1 = (
        "System: Kamu adalah asisten yang memberikan kesimpulan produk secara ringkas, objektif, dan alami berdasarkan data deskripsi dan sentimen.\n"
        "User: Buatkan kesimpulan apakah produk ini bagus dan worth it atau tidak, dengan gaya bahasa alami dan manusiawi. "
        "Gunakan informasi dari deskripsi dan ringkasan sentimen berikut.\n\n"
        f"Deskripsi produk:\n{description}\n\n"
        f"Ringkasan sentimen:\n{sentiment_summary}\n\n"
        "Kesimpulan:"
    )

    # Tokenisasi input dengan truncation dan max_length yang aman
    inputs1 = tokenizer_llm(prompt1, return_tensors="pt", truncation=True, max_length=1024).to(model_llm.device)

    # Generate output
    output1 = model_llm.generate(
        **inputs1,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        eos_token_id=tokenizer_llm.eos_token_id or tokenizer_llm.pad_token_id,  # fallback kalau EOS nggak ada
    )

    # Decode hasil
    full_output1 = tokenizer_llm.decode(output1[0], skip_special_tokens=True)
    # Ekstrak bagian setelah "Kesimpulan:"
    if "Kesimpulan:" in full_output1:
        conclusion = full_output1.split("Kesimpulan:")[-1].strip()
    else:
        conclusion = full_output1.strip()

    return conclusion


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

       #Koreksi prediksi sentimen berdasarkan confidence, rating, dan jumlah kata positif/negatif
        rating = review["Rating"]
        # if max_confidence < 0.6:  # Jika confidence rendah, buat netral
        #     sentiment = 1
        if sentiment == 0 and (rating == 3):
            sentiment = 1  # Ubah negatif jadi netral jika rating 3 atau lebih banyak kata positif
        elif sentiment == 2 and (rating == 3 ):
            sentiment = 1  # Ubah positif jadi netral jika rating 3 atau lebih banyak kata negatif
        elif sentiment == 0 and (rating in [4, 5] or positive_count > negative_count):
            sentiment = 1  # Ubah negatif jadi netral jika rating 4 atau 5 dengan cukup kata positif
        elif sentiment == 2 and (rating in [1, 2] or negative_count > positive_count):
            sentiment = 1  # Ubah positif jadi netral jika rating 1 atau 2 dengan cukup kata negatif
        elif sentiment == 1 and (rating in [4, 5] or positive_count > negative_count):
            sentiment = 2  # Ubah netral jadi positif jika rating 4 atau 5 dengan banyak kata positif
        elif sentiment == 1 and (rating in [1, 2] or negative_count > positive_count):
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
            reviews,desc= scrape_tokopedia_reviews(product_url, max_reviews=100)
        
        st.subheader("📌 Deskripsi Produk")
        st.markdown(desc)
        
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

             # Generate conclusion with DeepSeek
            with st.spinner("Membuat kesimpulan..."):
                conclusion = generate_conclusion_with_llm(desc, summary)
            
            st.subheader("📌 Kesimpulan")
            st.markdown(conclusion)

            sentiment_df.to_csv("Tokopedia_Reviews.csv", index=False)
            st.success("✅ Analisis selesai! Data ulasan disimpan sebagai 'Tokopedia_Reviews.csv'.")
    else:
        st.error("⚠️ Harap masukkan link produk Tokopedia yang valid!")
