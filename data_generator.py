import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"



def load_data():
    # Load iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    return df


def inject_drift(df):
    # Here we are simulating a sensor malfunction
    drifted_df = df.copy()
    drifted_df["sepal length (cm)"] = drifted_df["sepal length (cm)"] + 5.0
    return drifted_df


def main():
    # Load clean (no drift) data
    df = load_data()

    # Save normal dataset
    df.to_csv(DATA_DIR / "iris_train.csv", index=False)

    # Create and save drifted dataset
    drifted_df = inject_drift(df)
    drifted_df.to_csv(DATA_DIR / "iris_drifted.csv", index=False)

    # Plot histogram comparison
    plt.figure()
    plt.hist(df["sepal length (cm)"], bins=20, label="Normal")
    plt.hist(drifted_df["sepal length (cm)"], bins=20, label="Drifted")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Frequency")
    plt.title("Sepal Length Distribution: Normal vs Drifted")
    plt.legend()

    plt.savefig(BASE_DIR/"drift_plot.png")
    plt.close()


main()
