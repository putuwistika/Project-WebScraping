from waitress import serve
from flask import Flask, jsonify, request
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType
import numpy as np
from functools import lru_cache
import requests
import json
from flask_cors import CORS

app = Flask(__name__)

# Konfigurasi CORS
CORS(app, resources={
    r"/api/*": {"origins": ["https://hawk-enhanced-oyster.ngrok-free.app","http://localhost:5173", "https://frontend-product-comparison.vercel.app"]}
})

# UDF to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return float(dot_product / (norm_vec1 * norm_vec2))

# Convert the cosine similarity function into a PySpark UDF
cosine_similarity_udf = udf(cosine_similarity, FloatType())

# Inisialisasi SparkSession
spark = SparkSession.builder \
    .appName("DataPipeline") \
    .config("spark.jars", "postgresql-42.7.4.jar") \
    .getOrCreate()

# Mengatur properti koneksi ke PostgreSQL
url = "jdbc:postgresql://localhost:5432/db_product_padi"
properties = {
    "user": "putu",
    "password": "Dev!",
    "driver": "org.postgresql.Driver"
}

# Global variable to store cached data
cached_data = None

# Function to load all data from PostgreSQL into cache
def load_data_to_cache():
    global cached_data
    if cached_data is None:
        print("Loading data to cache...")
        df = spark.read.jdbc(url=url, table="product_data_clean", properties=properties) \
            .select("product_id", "product_name", "product_price", "stok", "jumlah_terjual", 
                    "rating", "image_srcset", "kategori", "brand", "min_pembelian", 
                    "berat_satuan", "dimensi_ukuran", "dikirim_dari", "penjual", 
                    "description") \
            .orderBy("product_id")  # Sort by product_id
        cached_data = df.toPandas().to_dict(orient="records")
    else:
        print("Cache already loaded.")

# Function to get full product details by product_id
def get_product_by_id(product_id):
    load_data_to_cache()  # Ensure cache is loaded before fetching data
    for product in cached_data:
        if product['product_id'] == product_id:
            return product
    return None

# Function to retrieve paginated data from cache
def get_paginated_products(limit=None, offset=0):
    load_data_to_cache()  # Ensure cache is loaded
    paginated_data = cached_data[offset:offset + limit]
    simplified_data = [
        {
            "product_id": product["product_id"],
            "product_name": product["product_name"],
            "product_price": product["product_price"],
            "stok": product["stok"],
            "jumlah_terjual": product["jumlah_terjual"],
            "rating": product["rating"],
            "image_srcset": product["image_srcset"],
            "kategori": product["kategori"]
        }
        for product in paginated_data
    ]
    return simplified_data

# Caching the function to improve performance
@lru_cache(maxsize=100)
def get_similar_products_cached(product_id):
    return get_similar_products(product_id)

# Function to get similar products using cosine similarity
def get_similar_products(product_id):
    df = spark.read.jdbc(url=url, table="product_data_clean", properties=properties)
    features_df_loaded = spark.read.parquet("features_df.parquet")

    # Cross join untuk menghitung kesamaan kosinus antara produk
    cross_df = features_df_loaded.alias('a').crossJoin(features_df_loaded.alias('b')) \
        .filter(col('a.product_id') == product_id) \
        .filter(col('a.product_id') != col('b.product_id'))

    # Hitung cosine similarity
    similarity_df = cross_df.withColumn("cosine_similarity", cosine_similarity_udf(col("a.features_vector"), col("b.features_vector")))

    # Gabungkan data similarity dengan detail produk
    final_df = similarity_df.join(df.alias('raw_data'), similarity_df['b.product_id'] == col('raw_data.product_id')) \
        .select('raw_data.product_id', 'raw_data.product_price', 'raw_data.product_name', 
                'raw_data.stok', 'raw_data.description', 'raw_data.jumlah_terjual', 'raw_data.rating', 
                'raw_data.image_srcset', 'raw_data.kategori', 'cosine_similarity')

    # Langkah 1: Urutkan berdasarkan cosine similarity tertinggi
    sorted_by_similarity_df = final_df.orderBy(col('cosine_similarity').desc())

    # Langkah 2: Urutkan kembali berdasarkan rating tertinggi dan jumlah_terjual terbanyak
    sorted_final_df = sorted_by_similarity_df.orderBy(col('cosine_similarity').desc(), col('rating').desc(), col('jumlah_terjual').desc())

    # Batasi hasil hanya ke 10 produk setelah pengurutan
    top_10_products = sorted_final_df.limit(10)

    # Ambil data produk
    result = top_10_products.collect()
    products = []
    for row in result:
        products.append({
            "product_id": row['product_id'],
            "product_name": row['product_name'],
            "product_price": row['product_price'],
            "stok": row['stok'],
            "description": row['description'],
            "jumlah_terjual": row['jumlah_terjual'],
            "rating": row['rating'],
            "image_srcset": row['image_srcset'],
            "kategori": row['kategori'],
            "cosine_similarity": row['cosine_similarity']
        })
    
    return products

from flask import Flask, send_file



# Menampilkan HTML dari file README_documentation.html ketika root URL diakses
@app.route('/')
def show_readme_html():
    # Path ke file HTML
    html_file_path = 'README_documentation.html'
    # Menggunakan send_file untuk mengirimkan file HTML ke browser
    return send_file(html_file_path)


# Route to get product similarity by product_id
@app.route('/api/product_similarity/<product_id>', methods=['GET'])
def product_similarity(product_id):
    try:
        products = get_similar_products_cached(product_id)
        return jsonify(products)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get paginated products (GET method)
@app.route('/api/products', methods=['GET'])
def get_products_get():
    try:
        limit = int(request.args.get("limit", 10))
        page = int(request.args.get("page", 1))
        offset = (page - 1) * limit
        data = get_paginated_products(limit=limit, offset=offset)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get paginated products (POST method)
@app.route('/api/products', methods=['POST'])
def get_products_post():
    try:
        request_data = request.get_json()
        limit = request_data.get("limit", 10)
        page = request_data.get("page", 1)
        offset = (page - 1) * limit
        data = get_paginated_products(limit=limit, offset=offset)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get product by product_id
@app.route('/api/products/<product_id>', methods=['GET'])
def get_product_by_id_route(product_id):
    try:
        product = get_product_by_id(product_id)
        if product:
            return jsonify(product)
        else:
            return jsonify({"error": "Product not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get product by product_id via POST
@app.route('/api/products_id', methods=['POST'])
def get_product_by_id_post():
    try:
        request_data = request.get_json()
        product_id = request_data.get("product_id")
        if not product_id:
            return jsonify({"error": "Missing product_id in the request body"}), 400
        
        product = get_product_by_id(product_id)
        if product:
            return jsonify(product)
        else:
            return jsonify({"error": "Product not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route GET untuk mengambil semua produk tanpa pagination
@app.route('/api/all_products', methods=['GET'])
def get_all_products():
    try:
        # Pastikan data sudah di-cache sebelum mengambil semua produk
        load_data_to_cache()
        
        # Mengambil semua produk tanpa pagination
        all_products = [
            {
                "product_id": product["product_id"],
                "product_name": product["product_name"],
                "product_price": product["product_price"],
                "stok": product["stok"],
                "jumlah_terjual": product["jumlah_terjual"],
                "rating": product["rating"],
                "image_srcset": product["image_srcset"],
                "kategori": product["kategori"]
            }
            for product in cached_data
        ]
        
        return jsonify(all_products)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to search products by name
def search_products_by_name(product_name):
    load_data_to_cache()  # Ensure cache is loaded
    # Filter cached data to only include products whose name contains the search term
    matching_products = [
        product for product in cached_data 
        if product['product_name'] and product_name.lower() in product['product_name'].lower()
    ]
    
    # Simplify the data returned to the user
    simplified_data = [
        {
            "product_id": product["product_id"],
            "product_name": product["product_name"],
            "product_price": product["product_price"],
            "stok": product["stok"],
            "jumlah_terjual": product["jumlah_terjual"],
            "rating": product["rating"],
            "image_srcset": product["image_srcset"],
            "kategori": product["kategori"]
        }
        for product in matching_products
    ]
    
    return simplified_data


# Route to search for products by name
@app.route('/api/search', methods=['GET'])
def search_by_product_name():
    try:
        # Get the product_name query parameter
        product_name = request.args.get("product_name", "")
        if not product_name:
            return jsonify({"error": "Product name is required"}), 400
        
        # Search products by name
        products = search_products_by_name(product_name)
        
        if products:
            return jsonify(products)
        else:
            return jsonify({"message": "No products found matching the search term."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API Configuration
LLM_API_BASE_URL = "https://api.groq.com/openai/v1"
LLM_API_KEY = "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"
LLM_CHAT_MODEL = "llama-3.1-8b-instant"
LLM_STREAMING = False

def query_groq(messages):
    url = f"{LLM_API_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    payload = {
        "model": LLM_CHAT_MODEL,
        "messages": messages,
        "max_tokens": 600,
        "temperature": 0.5,
        "stream": LLM_STREAMING,
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if not response.ok:
        raise Exception(f"HTTP error {response.status_code}: {response.text}")
    
    return response.json()

def create_comparison_prompt(products):
    prompt = (
        "KONSISTEN JIKA DITANYAN BERKALI KALI!, Kamu bernama Team3, jangan menggunakan kata Namun, Saya ingin Anda membandingkan beberapa produk berdasarkan atribut berikut: "
        "nama produk, rating, deskripsi, harga, jumlah terjual, kategori, minimal pembelian, "
        "dimensi ukuran, penjual, dan lokasi pengiriman. Fokuslah pada keunggulan, kekurangan, "
        "dan kecocokan pengiriman produk tersebut dengan pembeli. "
        "Ingat Buat dengan bahasa manusia dan singkat saja\n\n"
        "Berikut adalah produk-produk yang akan dibandingkan:\n"
    )
    
    for product in products:
        prompt += (
            f"Nama Produk: {product['product_name']}\n"
            f"Rating: {product['rating']}\n"
            f"Deskripsi: {product['description']}\n"
            f"Harga: Rp{product['product_price']}\n"
            f"Jumlah Terjual: {product['jumlah_terjual']}\n"
            f"Kategori: {product['kategori']}\n"
            f"Minimal Pembelian: {product['min_pembelian']}\n"
            f"Dimensi Ukuran: {product['dimensi_ukuran']}\n"
            f"Penjual: {product['penjual']}\n\n"
        )
    
    prompt += (
        "Berikut instruksi untuk hasil analisis YANG KONSISTEN:\n"
        "- Buat keunggulan dan kekurangan masing-masing produk berdasarkan atribut di atas.\n"
        "- Berikan kesimpulan produk mana yang lebih baik untuk pembeli tanpa imbuhan apapun.\n"
        "Fokus pada product ini saja, jangan hal lain, jangan menggunakan kata Namun. "
        "JANGAN MENGHINA/MENJELEKKAN PRODUCT MANAPUN, BAIK PENJUALNYA ATAUPUN PRODUCTNYA."
        "INGAT KALAU ANDA DITANYA 2 HINGGA BERKALI KALI, JAWABAN ANDA HARUS KONSISTEN"
        "Cara menjawab Berikan Hallo Digistar dan tetap 1 product yang lebih baik. "
        "Gunakan template berikut untuk menjawab:\n\n"
        "Hallo Digistar!\n\n"
        "Berikut adalah hasil analisis produk-produk yang dibandingkan:\n\n"
        "**Produk A**\n"
        "* Keunggulan:\n"
        "a.\n"
        "b.\n"
        "c.\n"
        "* Kekurangan:\n"
        "a.\n"
        "b.\n"
        "c.\n\n"
        "**Produk B**\n"
        "* Keunggulan:\n"
        "a.\n"
        "b.\n"
        "c.\n"
        "* Kekurangan:\n"
        "a.\n"
        "b.\n"
        "c.\n\n"
        "**Produk C**\n"
        "* Keunggulan:\n"
        "a.\n"
        "b.\n"
        "c.\n"
        "* Kekurangan:\n"
        "a.\n"
        "b.\n"
        "c.\n\n"
        "Saya rekomendasikan <Product Terbaik> Untuk anda beli Karena <Alasan>\n\n"
        "Terima Kasih, Jangan Lupa Checkout"
    )
    
    return [{"role": "user", "content": prompt}]

import re

def compare_products(products):
    prompt = create_comparison_prompt(products)
    response = query_groq(prompt)
    
    comparison_text = response['choices'][0]['message']['content']
    
    # Extract the recommendation line
    recommendation_line = re.findall(r"Saya rekomendasikan .* Untuk anda beli Karena .*", comparison_text)
    
    if not recommendation_line:
        return comparison_text, None
    
    recommendation_line = recommendation_line[0]
    
    # Create a mapping of product names to their IDs
    product_name_to_id = {product['product_name']: product['product_id'] for product in products}
    
    # Function to find the best matching product name
    def find_best_match(text, product_names):
        best_match = None
        max_words_matched = 0
        for name in product_names:
            words = set(name.lower().split())
            matched_words = sum(1 for word in words if word in text.lower())
            if matched_words > max_words_matched:
                max_words_matched = matched_words
                best_match = name
        return best_match

    # Find the best matching product name in the recommendation
    best_match = find_best_match(recommendation_line, product_name_to_id.keys())
    
    # Get the corresponding product ID
    best_product_id = product_name_to_id.get(best_match) if best_match else None
    
    return comparison_text, best_product_id

def fetch_products(product_ids):
    # Read data from PostgreSQL
    df = spark.read.jdbc(url=url, table="product_data_clean", properties=properties)
    
    # Filter the dataframe based on the given product_ids
    filtered_df = df.filter(col("product_id").isin(product_ids))
    
    # Convert the filtered dataframe to a list of dictionaries
    products = filtered_df.toPandas().to_dict('records')
    
    return products

@app.route('/api/ai-best-products', methods=['POST'])
def api_compare_products():
    data = request.json
    product_ids = data.get('product_ids', [])
    
    if not 2 <= len(product_ids) <= 3:
        return jsonify({"error": "Please provide 2 or 3 product IDs"}), 400
    
    products = fetch_products(product_ids)
    
    if len(products) != len(product_ids):
        return jsonify({"error": "One or more product IDs are invalid"}), 400
    
    comparison_result, best_product_id = compare_products(products)
    
    # Extract the recommended product name from the comparison result
    recommended_product_match = re.search(r"Saya rekomendasikan \*\*(.*?)\*\*", comparison_result)
    recommended_product_name = recommended_product_match.group(1) if recommended_product_match else None
    
    # Verify that the best_product_id matches the recommended product name
    if recommended_product_name:
        for product in products:
            if product['product_name'] == recommended_product_name:
                best_product_id = product['product_id']
                break
    
    return jsonify({
        "comparison_result": comparison_result,
        "best_product_id": best_product_id
    })

if __name__ == "__main__":
    serve(app, host='127.0.0.1', port=5000)
