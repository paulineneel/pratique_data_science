import pandas as pd
import yfinance as yf
import glob
import os
from google.colab import drive

def get_ratios():
    ratios = {
        "forwardPE": [], "beta": [], "priceToBook": [], "priceToSales": [],
        "dividendYield": [], "trailingEps": [], "debtToEquity": [],
        "currentRatio": [], "quickRatio": [], "returnOnEquity": [],
        "returnOnAssets": [], "operatingMargins": [], "profitMargins": []
    }

    companies = {
        "Baidu": "BIDU", "JD.com": "JD", "BYD": "BYDDY", "ICBC": "1398.HK",
        "Toyota": "TM", "SoftBank": "9984.T", "Nintendo": "NTDOY", "Hyundai": "HYMTF",
        "Reliance Industries": "RELIANCE.NS", "Tata Consultancy Services": "TCS.NS",
        "Apple": "AAPL", "Microsoft": "MSFT", "Amazon": "AMZN", "Alphabet": "GOOGL",
        "Meta": "META", "Tesla": "TSLA", "NVIDIA": "NVDA", "Samsung": "005930.KS",
        "Tencent": "TCEHY", "Alibaba": "BABA", "IBM": "IBM", "Intel": "INTC",
        "Oracle": "ORCL", "Sony": "SONY", "Adobe": "ADBE", "Netflix": "NFLX",
        "AMD": "AMD", "Qualcomm": "QCOM", "Cisco": "CSCO", "JP Morgan": "JPM",
        "Goldman Sachs": "GS", "Visa": "V", "Johnson & Johnson": "JNJ", "Pfizer": "PFE",
        "ExxonMobil": "XOM", "ASML": "ASML.AS", "SAP": "SAP.DE", "Siemens": "SIE.DE",
        "Louis Vuitton (LVMH)": "MC.PA", "TotalEnergies": "TTE.PA", "Shell": "SHEL.L"
    }

    for c in companies:
        ticker = yf.Ticker(companies[c])
        for r in ratios:
            ratios[r].append(ticker.info.get(r))

    df_ratios = pd.DataFrame(ratios)
    df_ratios.index = companies.keys()
    df_ratios.to_csv('df_ratios.csv', index=False)
    return df_ratios, companies

def mount_drive():
    drive.mount('/content/drive')

def historique(symbol, name, start, end):
    company_data = yf.download(symbol, start=start, end=end, progress=False)
    company_data.columns = company_data.columns.get_level_values(0)
    df = company_data[['Close']].copy()
    df.loc[:, 'Next Day Close'] = df['Close'].shift(-1)
    df.loc[:, 'Rendement'] = (df['Next Day Close'] - df['Close']) / df['Close']
    df.dropna(inplace=True)

    filename = f"/content/drive/MyDrive/Companies_historical_data/{name.replace(' ', '_')}_historical_data.csv"
    df.to_csv(filename)

def create_historical_data(companies):
    data_folder = "/content/drive/MyDrive/Companies_historical_data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    for name, symbol in companies.items():
        historique(symbol, name, '2019-01-01', '2024-01-01')

def merge_all_csv():
    data_folder = "/content/drive/MyDrive/Companies_historical_data"
    csv_files = glob.glob(f"{data_folder}/*_historical_data.csv")
    all_data = []

    for file in csv_files:
        df = pd.read_csv(file)
        company_name = os.path.basename(file).split('_historical_data.csv')[0].replace('_', ' ')
        df["Company"] = company_name
        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)
    df_all.to_csv("Companies_historical_data.csv", index=False)
    return df_all


if __name__ == "__main__":
    print("🔁 Démarrage du TP1")

    # Étape 1 : monter Google Drive
    mount_drive()

    # Étape 2 : récupérer ratios et écrire df_ratios.csv
    df_ratios, companies = get_ratios()
    print("✅ Ratios enregistrés dans df_ratios.csv")

    # Étape 3 : télécharger l'historique et sauvegarder les CSV
    create_historical_data(companies)
    print("✅ Historique sauvegardé dans Companies_historical_data/")

    # Étape 4 : fusionner tous les fichiers CSV
    df_all = merge_all_csv()
    print("✅ Fichier fusionné : Companies_historical_data.csv")

    print("🎉 TP1 terminé avec succès.")
