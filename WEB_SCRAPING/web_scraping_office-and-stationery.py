import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)

# Set up the SparkSession
spark = SparkSession.builder \
    .appName("PostgreSQL Connection with PySpark") \
    .config("spark.jars", "postgresql-42.7.4.jar") \
    .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=log4j.properties") \
    .getOrCreate()

# PostgreSQL connection properties
jdbc_url = "jdbc:postgresql://localhost:5432/db_product_padi"
properties = {
    "user": "putu",
    "password": "Dev!",
    "driver": "org.postgresql.Driver"
}

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--disable-gpu")  # Disable GPU rendering
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems


# Define schema for the DataFrame
schema = StructType([
    StructField("Product_Name", StringType(), True),
    StructField("Product_Price", StringType(), True),
    StructField("Stok", StringType(), True),
    StructField("Jumlah_Terjual", StringType(), True),
    StructField("Rating", StringType(), True),
    StructField("Image_Srcset", StringType(), True),
    StructField("Kategori", StringType(), True),
    StructField("Brand", StringType(), True),
    StructField("Min_Pembelian", StringType(), True),
    StructField("Berat_Satuan", StringType(), True),
    StructField("Dimensi_Ukuran", StringType(), True),
    StructField("Dikirim_Dari", StringType(), True),
    StructField("Penjual", StringType(), True),
    StructField("Description", StringType(), True),
    StructField("URL", StringType(), True),
    StructField("Update", TimestampType(), True)
])

# Set up the WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

product_count = 0  # Initialize counter for product data


try:
    # Iterate through pages
    for page in range(1, 201):  # Page range from 1 to 100
        url = f'https://padiumkm.id/c/office-and-stationery?page={page}'
        driver.get(url)

        # Wait for the page to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

        # Parse the search results page
        soup = BeautifulSoup(driver.page_source, 'lxml')

        # Find all product links
        product_links = soup.find_all('a', href=True)

        # Filter product links based on the pattern you described
        filtered_links = [link['href'] for link in product_links if '/product/' in link['href']]

        # Loop through each product link
        for href in filtered_links:
            try:
                # Navigate to the product page
                driver.get(f'https://padiumkm.id{href}')

                # Wait for the product page to load
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

                # Parse the product page
                product_soup = BeautifulSoup(driver.page_source, 'lxml')

                # Extract product image srcset
                product_images_div = product_soup.find('div', id='product-images')
                srcset_value = product_images_div.find('img', srcset=True)['srcset'] if product_images_div else None

                # Extract product name
                product_name_tag = product_soup.find('h1', class_='text-sm md:text-base font-medium text-paletteText-primary font-ubuntu mb-1 capitalize')
                product_name = product_name_tag.get_text(strip=True) if product_name_tag else None

                # Extract product price
                product_price_tag = product_soup.find('label', class_='font-bold text-base md:text-2xl text-paletteText-primary font-ubuntu')
                product_price = product_price_tag.get_text(strip=True) if product_price_tag else None

                # Extract jumlah terjual
                jumlah_terjual_tag = product_soup.find('div', class_='text-sm flex-nowrap text-paletteText-primary font-ubuntu')
                jumlah_terjual = jumlah_terjual_tag.get_text(strip=True) if jumlah_terjual_tag else None

                # Extract rating
                rating_tag = product_soup.find('div', class_='text-xs flex-nowrap text-paletteText-primary font-ubuntu font-medium pr-1')
                rating = rating_tag.get_text(strip=True) if rating_tag else None

                # Extract asal dikirim
                asal_dikirim_tag = product_soup.find('div', class_='text-paletteText-primary text-sm font-[500]')
                asal_dikirim = asal_dikirim_tag.get_text(strip=True) if asal_dikirim_tag else None

                # Extract seller information
                seller_info = product_soup.find('div', class_='flex-1 flex-col')
                penjual = seller_info.find('a').find('span', class_='text-base font-semibold block text-paletteText-primary cursor-pointer h-[24px] overflow-hidden').get_text(strip=True) if seller_info else None

                # Extract product description
                deskripsi_container = product_soup.find('div', class_='space-y-3 px-0 text-paletteText-primary font-ubuntu')
                deskripsi = deskripsi_container.find('p', class_='break-words whitespace-pre-line w-full').get_text(strip=True) if deskripsi_container else None

                # Extract stock information
                stock_info = soup.find('div', class_='flex items-center text-paletteText-inactive text-sm')
                stok = stock_info.find_all('span', class_='text-[#444B55]')[1].get_text(strip=True) if stock_info else None

                # Find all rows (tr) in the tbody
                rows = product_soup.find_all('tr')

                # Initialize variables for each detail
                kategori, brand, min_pembelian, berat_satuan, dimensi_ukuran = None, None, None, None, None

                for row in rows:
                    try:
                        label = row.find('div', class_='text-[#8C9197]').get_text(strip=True)
                        value = row.find('div', class_='font-[500]').get_text(strip=True)

                        if label == 'Kategori':
                            kategori = value
                        elif label == 'Brand':
                            brand = value
                        elif label == 'Min Pembelian':
                            min_pembelian = value
                        elif label == 'Berat Satuan':
                            berat_satuan = value
                        elif label == 'Dimensi Ukuran':
                            dimensi_ukuran = value + ' ' + row.find_all('div', class_='font-[500]')[1].get_text(strip=True)
                    except Exception as e:
                        print(f"Error processing row: {e}")

                # Get the current timestamp in Python
                current_timestamp = datetime.now()

                # Create a DataFrame for the single row of data
                single_data_df = spark.createDataFrame(
                    [(product_name, product_price, stok, jumlah_terjual, rating, srcset_value, kategori, brand, min_pembelian, berat_satuan, dimensi_ukuran, asal_dikirim, penjual, deskripsi, f'https://padiumkm.id{href}', current_timestamp)],
                    schema=schema
                )

                # Write data to PostgreSQL in real-time
                single_data_df.write.jdbc(url=jdbc_url, table="product_data", mode="append", properties=properties)

                product_count += 1  # Increment the product counter

                # Print the log message
                print(f"data ke - {product_count} ({penjual if penjual else 'Unknown'}) berhasil tersimpan")

            except Exception as e:
                print(f"Error processing product {href}: {e}")

            # Go back to the previous page (search results)
            driver.back()

            # Wait for the search results page to load again
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

            # Pause for a moment to avoid overwhelming the server (optional)
            time.sleep(1)

        # Pause between pages
        time.sleep(3)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the browser
    driver.quit()

