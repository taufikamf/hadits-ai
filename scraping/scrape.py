import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def scrape_hadith(collection, number):
    url = f"https://www.hadits.id/hadits/{collection}/{number}"
    print(f"Mengambil data dari: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error saat melakukan request ke {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    hadith_content = soup.find('article', class_='hadits-content')
    if not hadith_content:
        print(f"Tidak dapat menemukan konten hadits di {url}. Hadits mungkin tidak ada atau struktur HTML berubah.")
        return None

    arabic_text_tag = hadith_content.find('p', class_='rtl')
    arabic_text = arabic_text_tag.get_text(strip=True) if arabic_text_tag else "Teks Arab Tidak Ditemukan"

    terjemah_text_tag = None
    if arabic_text_tag:
        terjemah_text_tag = arabic_text_tag.find_next_sibling('p')

    terjemah_text = terjemah_text_tag.get_text(strip=True) if terjemah_text_tag else "Terjemahan Tidak Ditemukan"

    hadith_data = {
        'id': number,
        'kitab': collection,
        'arab': arabic_text,
        'terjemah': terjemah_text,
    }
    return hadith_data

if __name__ == "__main__":
    koleksi_hadits = 'bukhari'
    max_hadith_number = 50

    print(f"Memulai scraping {max_hadith_number} hadits dari koleksi {koleksi_hadits}...\n")

    hadits_list = []
    for i in range(1, max_hadith_number + 1):
        hadith = scrape_hadith(koleksi_hadits, i)
        if hadith:
            hadits_list.append(hadith)
        else:
            print(f"Tidak dapat mengambil hadits {koleksi_hadits}/{i}. Menghentikan scraping untuk koleksi ini.\n")
            break

