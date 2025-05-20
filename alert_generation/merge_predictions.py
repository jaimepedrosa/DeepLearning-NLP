import pandas as pd
from pathlib import Path

def merge_predictions(sa_path, ner_path, output_path):
    sa_df = pd.read_csv(sa_path)
    ner_df = pd.read_csv(ner_path)
    merged_df = pd.merge(sa_df, ner_df, on="Sentence", suffixes=("_sa", "_ner"))
    merged_df.to_csv(output_path, index=False)
    print(f"Guardado en {output_path}")
if __name__ == "__main__":
    base_sa = Path("sentimentAnalysis/csvs")
    base_ner = Path("ner/results")
    merge_predictions(base_sa / "train_predictions.csv", base_ner / "train_predictions.csv", "alert_generation/outputs/merged_train.csv")
    merge_predictions(base_sa / "val_predictions.csv", base_ner / "val_predictions.csv", "alert_generation/outputs/merged_val.csv")
    #merge_predictions(base_sa / "test_predictions.csv", base_ner / "test_predictions.csv", "alertGeneration/outputs/merged_test.csv")
