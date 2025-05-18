
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import shap
import pandas as pd
import glob
import ta
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

results = []

def load_and_engineer_features():
    filepaths = glob.glob("Companies_historical_data/*.csv")
    liste_dataframes = []
    for filepath in filepaths:
        df = pd.read_csv(filepath)[['Close']]
        df['Close Horizon'] = df['Close'].shift(-20)
        df['Horizon Return'] = (df['Close Horizon'] - df['Close']) / df['Close']
        df['label'] = df['Horizon Return'].apply(lambda x: 2 if x > 0.05 else (0 if x < -0.05 else 1))
        df['SMA 20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['EMA 20'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['RSI 14'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd(df['Close'])
        df['MACD Signal'] = ta.trend.macd_signal(df['Close'])
        df['Bollinger High'] = ta.volatility.bollinger_hband(df['Close'], window=20)
        df['Bollinger Low'] = ta.volatility.bollinger_lband(df['Close'], window=20)
        df['Rolling Volatility 20'] = df['Close'].rolling(window=20).std()
        df['ROC 10'] = ta.momentum.roc(df['Close'], window=10)
        liste_dataframes.append(df)
    base_donnees = pd.concat(liste_dataframes, ignore_index=True)
    return base_donnees

def preprocess_data(base_donnees):
    y = base_donnees['label']
    X = base_donnees.drop(columns=['label', 'Close Horizon', 'Weekly return', 'Next Day Close'], errors='ignore')
    X.dropna(inplace=True)
    y = y[X.index]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y

def split_data(X_scaled, y):
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def run_model(name, model, param_grid, X_train, y_train, X_test, y_test, shap_type='tree'):
    print(f"\nTraining model: {name}")
    clf = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"\n{name} - Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    shap.initjs()
    if shap_type == 'tree':
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values[2], X_test, feature_names=X.columns, show=False)
        plt.title(f"{name} - SHAP Summary (Buy)")
        plt.show()
        shap.summary_plot(shap_values[0], X_test, feature_names=X.columns, show=False)
        plt.title(f"{name} - SHAP Summary (Sell)")
        plt.show()
    elif shap_type == 'kernel':
        sample = pd.DataFrame(X_test).sample(100, random_state=0).reset_index(drop=True)
        background = pd.DataFrame(X_train).sample(100, random_state=0).reset_index(drop=True)
        explainer = shap.KernelExplainer(best_model.predict_proba, background)
        shap_values = explainer.shap_values(sample)
        shap.summary_plot(shap_values[2], sample, show=False)
        plt.title(f"{name} - SHAP Summary (Buy)")
        plt.show()
        shap.summary_plot(shap_values[0], sample, show=False)
        plt.title(f"{name} - SHAP Summary (Sell)")
        plt.show()
    results.append({
        'Model': name,
        'Best Params': clf.best_params_,
        'Train Score': best_model.score(X_train, y_train),
        'Test Score': best_model.score(X_test, y_test),
        'Precision (Buy)': report.get('2', {}).get('precision'),
        'Recall (Buy)': report.get('2', {}).get('recall'),
        'F1 (Buy)': report.get('2', {}).get('f1-score'),
        'Precision (Hold)': report.get('1', {}).get('precision'),
        'Recall (Hold)': report.get('1', {}).get('recall'),
        'F1 (Hold)': report.get('1', {}).get('f1-score'),
        'Precision (Sell)': report.get('0', {}).get('precision'),
        'Recall (Sell)': report.get('0', {}).get('recall'),
        'F1 (Sell)': report.get('0', {}).get('f1-score'),
        'F1 (Macro Avg)': report.get('macro avg', {}).get('f1-score')
    })

def run_all_models(X_train, y_train, X_test, y_test):
    run_model('Random Forest', RandomForestClassifier(),
              {'n_estimators': [100, 200], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]},
              X_train, y_train, X_test, y_test, 'tree')

    run_model('XGBoost', XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
              {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]},
              X_train, y_train, X_test, y_test, 'tree')

    run_model('KNN', KNeighborsClassifier(),
              {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
              X_train, y_train, X_test, y_test, 'kernel')

    run_model('Logistic Regression', LogisticRegression(max_iter=1000),
              {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs'], 'multi_class': ['multinomial']},
              X_train, y_train, X_test, y_test, 'kernel')

    run_model('SVM', SVC(probability=True),
              {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
              X_train, y_train, X_test, y_test, 'kernel')

def display_results():
    df = pd.DataFrame(results)
    print("\nRésumé des performances :")
    print(df.sort_values(by='F1 (Macro Avg)', ascending=False))
    return df
