import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
import csv
import re

## Baslik yazildiginda o baslik adına csv dosyası oluştur ve entry'leri kaydet.
# url adresini belirle

url = r"https://www.sahibinden.com/minivan-panelvan-mitsubishi-l-300?pagingOffset=350&pagingSize=50"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36"}
r = requests.get(url, headers=headers)
soup = bs(r.text,"html.parser")
page_numbers = soup.select("[id=currentPageValue]")[0]['value']


crawling_url = r"https://www.sahibinden.com/minivan-panelvan-mitsubishi-l-300?pagingOffset="

url_list = [crawling_url+str(x)+"&pagingSize=50" for x in range(0,int(page_numbers)*50,50)]



def get_entry_per_page():
    with open("data.csv", "w", newline='', encoding="utf-8") as csvfile:
        header_list = ["fiyat","model","yıl","KM","ilan_tarihi","il","ilçe"]
        writer = csv.writer(csvfile)
        writer.writerow(header_list)
        
        for i in url_list:

            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36"}
            
            # adrese istek at
            
            r = requests.get(i, headers=headers)
            
            # sayfadaki texti beautifulsoup ile parse ettir
            
            soup = bs(r.text,"html.parser")
            
            # istenen değişkenlerin bulunduğu kısımları seç
            
            fiyat = soup.select("[class~=searchResultsPriceValue]")
            model = soup.select("[class~=searchResultsTagAttributeValue]")
            yıl = soup.select("[class~=searchResultsAttributeValue]")
            KM = soup.select("[class~=searchResultsAttributeValue]")
            ilan_tarihi  = soup.select("[class~=searchResultsDateValue]")
            il  = soup.select("[class~=searchResultsLocationValue]")
            ilçe = soup.select("[class~=searchResultsLocationValue]")
            

            # bulunduğu kısımlardan text'leri al

            for j in range(50):
                writer.writerow([
                    fiyat[j].get_text()[2:-3],
                    model[1].get_text()[25:],
                    yıl[3*j].get_text()[21:],
                    KM[3*j+1].get_text()[21:],
                    ilan_tarihi[0].get_text("span").split("span")[1]+" "+ilan_tarihi[0].get_text("span").split("span")[4],
                    il[j].get_text("br").split("br")[0][25:],
                    ilçe[j].get_text("br").split("br")[1]]
                    )

get_entry_per_page()