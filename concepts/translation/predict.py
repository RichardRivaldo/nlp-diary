import json

import pandas as pd
import requests


def load_data(path):
    with open(path) as f:
        lst = [tuple(x) for x in json.load(f)]
        return lst


def predict(row):
    req = {"input_sentence": row}
    resp = requests.post("http://localhost:8000/translation", req).json()
    resp = resp["data"]["translated"]
    return resp


loaded_train = load_data("train.json")
loaded_val = load_data("val.json")
loaded_test = load_data("test.json")

test_df = pd.DataFrame(loaded_test, columns=["input", "label"])
test_df["predicted"] = test_df["input"].map(predict)
test_df.to_csv("result.csv")
