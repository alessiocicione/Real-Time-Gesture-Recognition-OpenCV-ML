import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def train_and_save():
    train_df = pd.read_csv('datasets/regenerated/gesture_train_dataset_4.csv')
    test_df = pd.read_csv('datasets/regenerated/gesture_test_dataset_4.csv')

    drop_columns = ['coord_0000_X', 'coord_0000_Y', 'origin_dist_0000', 'is_visible_0000',
                    'd_1100_0101', 'd_1010_1100', 'd_0011_1010', 'd_1111_0011']
    train_df = train_df.drop(columns=drop_columns)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.drop(columns=drop_columns)

    target_column = 'reading_id'
    data_train = train_df.drop(columns=[target_column])
    target_train = train_df[target_column]
    data_test = test_df.drop(columns=[target_column])
    target_test = test_df[target_column]

    scaler = MinMaxScaler()
    data_train_normalized = scaler.fit_transform(data_train)
    data_test_normalized = scaler.transform(data_test)

    # === SVM con iperparametri ottimizzati ===
    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(data_train_normalized, target_train)
    y_pred_svm = svm.predict(data_test_normalized)
    accuracy_svm = accuracy_score(target_test, y_pred_svm)
    print(f"Accuratezza SVM: {accuracy_svm:.4f}")

    # === Salva modello e scaler ===
    joblib.dump(svm, 'pybindsave/svm_model_new.joblib')
    joblib.dump(scaler, 'pybindsave/scaler_new.joblib')
    print("Modello SVM e scaler salvati.")

if __name__ == "__main__":
    train_and_save()
