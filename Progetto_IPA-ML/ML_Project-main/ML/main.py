import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # === Carico i dataset ===
    train_df = pd.read_csv('datasets/regenerated/gesture_train_dataset_4.csv')
    test_df = pd.read_csv('datasets/regenerated/gesture_test_dataset_4.csv')

    # === Colonne da escludere (dati geometrici del riferimento, distanze duplicate) ===
    drop_columns = ['coord_0000_X', 'coord_0000_Y', 'origin_dist_0000', 'is_visible_0000',
                    'd_1100_0101', 'd_1010_1100', 'd_0011_1010', 'd_1111_0011']
    train_df = train_df.drop(columns=drop_columns)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle del test set
    test_df = test_df.drop(columns=drop_columns)

    # === Preparazione features e target ===
    target_column = 'reading_id'
    data_train = train_df.drop(columns=[target_column])
    target_train = train_df[target_column]
    data_test = test_df.drop(columns=[target_column])
    target_test = test_df[target_column]

    # === Scaler ===
    #scaler = StandardScaler()
    #scaler = RobustScaler()
    scaler = MinMaxScaler()

    # === Grid Search KNN ===
    print("--- Grid Search per KNN ---")
    knn_pipeline = Pipeline([
        ('scaler', scaler),
        ('knn', KNeighborsClassifier())
    ])
    knn_params = {
        'knn__n_neighbors': [3, 5, 7, 9],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]
    }
    grid_knn = GridSearchCV(knn_pipeline, knn_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_knn.fit(data_train, target_train)
    y_pred_knn = grid_knn.predict(data_test)
    print("Best KNN parameters:", grid_knn.best_params_)
    print(f"KNN Accuracy: {accuracy_score(target_test, y_pred_knn):.4f}\n")

    # === Grid Search Random Forest ===
    print("--- Grid Search per Random Forest ---")
    rf_pipeline = Pipeline([
        ('scaler', scaler),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    rf_params = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2],
        'rf__max_features': ['sqrt', 'log2']
    }
    grid_rf = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_rf.fit(data_train, target_train)
    y_pred_rf = grid_rf.predict(data_test)
    print("Best RF parameters:", grid_rf.best_params_)
    print(f"Random Forest Accuracy: {accuracy_score(target_test, y_pred_rf):.4f}\n")

    # === Grid Search SVM ===
    print("--- Grid Search per SVM ---")
    svm_pipeline = Pipeline([
        ('scaler', scaler),
        ('svm', SVC(probability=True))
    ])
    svm_params = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 0.01, 0.1],
        'svm__kernel': ['rbf', 'linear']
    }
    grid_svm = GridSearchCV(svm_pipeline, svm_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_svm.fit(data_train, target_train)
    y_pred_svm = grid_svm.predict(data_test)
    print("Best SVM parameters:", grid_svm.best_params_)
    print(f"SVM Accuracy: {accuracy_score(target_test, y_pred_svm):.4f}\n")

    # === Report Classificazione ===
    print('Classification Report - Random Forest')
    print(classification_report(target_test, y_pred_rf))
    print('Classification Report - KNN')
    print(classification_report(target_test, y_pred_knn))
    print('Classification Report - SVM')
    print(classification_report(target_test, y_pred_svm))

    # === Matrici di confusione ===
    for model_name, y_pred, cmap, target in [
        ("Random Forest", y_pred_rf, 'Blues', target_test),
        ("KNN", y_pred_knn, 'Greens', target_test),
        ("SVM", y_pred_svm, 'Oranges', target_test)
    ]:
        print(f"Confusion matrix {model_name}")
        cm = confusion_matrix(target, y_pred, normalize='true')
        print(cm)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()