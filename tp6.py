import requests
import json
from datetime import datetime, timedelta
import os
import glob

# Clé API à récuperer sur https://newsapi.org
api_key = "17dbe9704d5b425aa3b0224313457fef"

# sources d'actualités financieres
sources = 'financial-post,the-wall-street-journal,bloomberg,the-washington-post,australian-financial-review,bbc-news,cnn'


# on initialise les params pour appeler l'API
def initialize_params(company_name: str):
    last_day = datetime.today().strftime('%Y-%m-%d')
    first_day = (datetime.today() - timedelta(days=20)).strftime('%Y-%m-%d')

    return {
        "sources": sources,
        "q": company_name,
        "apiKey": api_key,
        "language": "en",
        "pageSize": 100,
        "from": first_day,
        "to": last_day,
    }

# on sauvergarde les actualités récupérées dans un fihcier json
def save_news(news_dict: dict, company_name: str, directory: str):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{company_name.lower()}_news.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(news_dict, f, indent=4, ensure_ascii=False)

# on récupère les actualité d'unne entreprise donnée
def get_news(company_name: str, path: str):
    url = 'https://newsapi.org/v2/everything'
    params = initialize_params(company_name)
    response = requests.get(url, params=params)

    if response.status_code == 200:
        news_data = response.json()
        news_dict = {}

        for article in news_data.get('articles', []):
            title = article.get('title')
            description = article.get('description')
            published_at = article.get('publishedAt', '').split("T")[0]
            source_name = article.get('source', {}).get('name')
            url = article.get('url')

            if title and description and (company_name.lower() in title.lower() or company_name.lower() in description.lower()):
                news_dict.setdefault(published_at, []).append({
                    'title': title,
                    'description': description,
                    'publishedAt': published_at,
                    'source': source_name,
                    'url': url
                })

        save_news(news_dict, company_name, path)
        return news_dict
    else:
        print(f"Erreur lors de la récupération des actualités : {response.status_code}")
        return {}

# on charge les actualités déjà enregistrées
def load_existing_news(company_name: str, directory: str = "news_data") -> dict:
    filepath = os.path.join(directory, f"{company_name.lower()}_news.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# on met à jour les actualités pour une entreprise
def update_news(company_name: str, file_path: str):
    existing_news = load_existing_news(company_name, file_path)
    new_news = get_news(company_name, file_path)

    for date, articles in new_news.items():
        existing_news.setdefault(date, [])
        for article in articles:
            if not any(existing_article['title'] == article['title'] for existing_article in existing_news[date]):
                existing_news[date].append(article)

    os.makedirs(file_path, exist_ok=True)
    output_file = os.path.join(file_path, f"{company_name.lower()}_news.json")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(existing_news, file, indent=4, ensure_ascii=False)
    print(f"News updated and saved to {output_file}")


# on récupére les actualités pour toutes les entreprises listées dans des fichiers CSV
def get_news_all_companies(new_data_path: str, company_name_path: str):
    filepaths = glob.glob(f"{company_name_path}/*.csv")

    for file in filepaths:
        filename = os.path.basename(file)
        company_name = os.path.splitext(filename)[0].split('_')[0]
        get_news(company_name, new_data_path)

if __name__ == "__main__":
    companies_name_path = "Companies_historical_data"
    new_data_path = "Companies_news_data"
    get_news_all_companies(new_data_path, companies_name_path)
