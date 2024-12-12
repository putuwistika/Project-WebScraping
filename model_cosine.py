from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.sql.functions import col
import numpy as np

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

# Baca data dari PostgreSQL
df = spark.read.jdbc(url=url, table="product_data", properties=properties)
    
# Membaca kembali DataFrame dari format Parquet
features_df_loaded = spark.read.parquet("features_df.parquet")

# Misalkan kita ingin menguji rekomendasi untuk produk dengan product_id tertentu, misalnya '12345'
target_product_id = "8365736d-599d-4489-a262-8dc270afeba6"


# Self-join dan filter hanya menghitung kemiripan untuk produk target berdasarkan product_id
cross_df = features_df_loaded.alias('a').crossJoin(features_df_loaded.alias('b')) \
    .filter(col('a.product_id') == target_product_id) \
    .filter(col('a.product_id') != col('b.product_id'))

# Menghitung cosine similarity antara vektor fitur dari produk target dan semua produk lainnya
similarity_df = cross_df.withColumn("cosine_similarity", cosine_similarity_udf(col("a.features_vector"), col("b.features_vector")))

# Menampilkan produk dengan kemiripan tertinggi ke produk target (hanya product_id dan similarity dulu)
top_similar_products = similarity_df.orderBy(col("cosine_similarity").desc()) \
    .select('b.product_id', 'cosine_similarity').limit(10)

# Join dengan data raw (df) untuk mendapatkan product_name
final_df = top_similar_products.join(df.alias('raw_data'), top_similar_products['product_id'] == col('raw_data.product_id')) \
    .select('raw_data.product_id','raw_data.product_price', 'raw_data.product_name', 'cosine_similarity')

# Menampilkan hasil
final_df.show(truncate=False)
