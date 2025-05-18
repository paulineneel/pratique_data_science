import tp1   
import tp2   
import tp3 
import tp4   
import tp5   
import tp6  
import tp7 
import tp8 

import os
import pandas as pd

def get_companies_list():
    return [
        "Apple", "Microsoft", "Amazon", "Alphabet", "Meta", "Tesla", "NVIDIA"
    ] 
  
def pipeline():
    print("===== PIPELINE D'ANALYSE COMPLET =====")
    companies = get_companies_list()
    results = []

    for company in companies:
        print(f"\n--- Analyse pour {company} ---")

        # on laisse les fonctions d'appels en commentaires 
        # données financières
        print("Chargement des ratios et historique...")
        # tp1.fetch_and_save_ratios(company)  # À adapter
        # tp1.download_historical_data(company)

        # clustering
        print("Clustering pour profil de risque et rendement...")
        # similar_companies = tp2.find_similar_companies(company)

        # ML
        print("Classification buy/hold/sell...")
        # prediction_ml = tp4.predict_company_label(company)

        # régression
        print("Prédiction de rendement à J+1...")
        # predicted_return = tp5.predict_return(company)

        # news
        print("Scraping & sentiment...")
        # news = tp6.get_news(company)
        # sentiments = tp8.analyze_sentiment(news)

        # décision
        print("Agrégation des signaux pour décision finale...")
        decision = "buy"  

        results.append({
            "Entreprise": company,
            "Recommandation": decision,
            # "Similaires": similar_companies,
            # "Rendement Prévu J+1": predicted_return,
            # "Sentiment Global": sentiments["global"],
            # "News": news["titles"]
        })

    # Sauvegarde finale
    df_resultats = pd.DataFrame(results)
    output_path = "resultats_recommandations.csv"
    df_resultats.to_csv(output_path, index=False)
    print(f"\nRésultats sauvegardés dans {output_path}")

if __name__ == "__main__":
    pipeline()
