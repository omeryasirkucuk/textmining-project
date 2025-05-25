import json
import time
from bs4 import BeautifulSoup
import requests
start_time = time.time()

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"}

with open('recipes/urls.json', 'r') as file:
    urls = json.load(file)

with open('recipes/detailed_recipes.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

url_ids = {item['id'] for item in urls}
data_ids = {item['id'] for item in data}
missing_ids = url_ids - data_ids
missing_urls = {item['id']: item['url'] for item in urls if item['id'] in missing_ids}

try:
    with open('recipes/detailed_recipes.json', 'r', encoding='utf-8') as file:
        existing_data = json.load(file)
except (FileNotFoundError, json.decoder.JSONDecodeError):
    existing_data = []

recipe_list = existing_data if isinstance(existing_data, list) else []

for id, url in missing_urls.items():
    try:
        # title
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html")
        title = soup.find("h1", {"class": "d-inline"}).text.strip()

        # image
        content = soup.find_all("div", {"class": "ContentRecipe_featuredImageWrapper__8truK"})
        img_tag = content[0].find("img").attrs['src']

        # categorical
        content = soup.find("div", {"class": "Breadcrumb_breadcrumbs__ZnTLV"})
        categories = []
        for item in content.find_all("a"):
            categories.append(item.text.strip())
        title = content.find("span", {"class": "Breadcrumb_contentTitle__8VL4A"}).text.strip()
        categories.append(title)

        # boyut, hazirlama suresi, pisirme suresi
        content = soup.find_all("div", {"class": "ContentRecipe_recipeDetail__0EBU0"})
        for item in content:
            h3_tag = item.find("h3")
            if h3_tag:
                if h3_tag.text.strip() == "KAÇ KİŞİLİK":
                    size = item.find("span").text.strip()
                elif h3_tag.text.strip() == "HAZIRLAMA SÜRESİ":
                    preparing_time = item.find("span").text.strip()
                elif h3_tag.text.strip() == "PİŞİRME SÜRESİ":
                    cooking_time = item.find("span").text.strip()

        # ingredient
        content = soup.find("ul", {"class": "Ingredients_ingredientList__DhBO1"})
        ingredients = []
        for li in soup.find_all('li'):
            spans = li.find_all('span')
            if len(spans) >= 3:
                quantity = spans[0].text.strip()
                unit = spans[1].text.strip()
                name = spans[2].text.strip()
                ingredients.append(f"{quantity} {unit} {name}")

        json_dict = {
            'id': id,
            'url': url,
            'title': title,
            'img_tag': img_tag,
            'categories': categories,
            'size': size,
            'preparing_time': preparing_time,
            'cooking_time': cooking_time,
            'ingredients': ingredients
        }

        recipe_list.append(json_dict)
        with open('recipes/detailed_recipes.json', 'w', encoding='utf-8') as file:
            json.dump(recipe_list, file, indent=4, ensure_ascii=False)
        print(f"{url} - {id} done")
    except Exception as e:
        print(e)