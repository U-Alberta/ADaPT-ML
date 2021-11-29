import sys
import requests
import json
import pandas as pd

pd.set_option('display.max_columns', None)

MULTICLASS_RESPONSE = {
    "table_name": [
        "example_data",
        "example_data",
        "example_data",
        "example_data",
        "example_data"
    ],
    "id": [
        "22",
        "18",
        "24",
        "19",
        "15"
    ],
    "cat": [
        0,
        0,
        0,
        0,
        0
    ],
    "dog": [
        0,
        1,
        0,
        1,
        0
    ],
    "bird": [
        1,
        0,
        0,
        0,
        0
    ],
    "horse": [
        0,
        0,
        0,
        0,
        0
    ],
    "snake": [
        0,
        0,
        1,
        0,
        1
    ],
    "prob_cat": [
        0.0000026098141376293395,
        5.737145693467853e-7,
        8.733156590178686e-10,
        0.000019467044194134946,
        0.0000027121710148853436
    ],
    "prob_dog": [
        0.000007023715529254579,
        0.9998007380917788,
        3.2998484966909616e-7,
        0.9999241130386705,
        0.00019959738566998405
    ],
    "prob_bird": [
        0.999838680754647,
        1.8773527537233474e-7,
        2.0500919041231248e-7,
        6.759103255872489e-7,
        0.00002999751685197958
    ],
    "prob_horse": [
        0.000014004118355031855,
        2.6298438151911457e-8,
        8.386656308045074e-10,
        2.701185355543245e-7,
        0.0000016878180401048688
    ],
    "prob_snake": [
        0.00013768159733110625,
        0.00019847415993833877,
        0.9999994632939787,
        0.00005547388827420121,
        0.9997660051084231
    ]
}
MULTILABEL_RESPONSE = {
    "table_name": [
        "example_data",
        "example_data",
        "example_data",
        "example_data",
        "example_data"
    ],
    "id": [
        "12",
        "05",
        "03",
        "08",
        "06"
    ],
    "cat": [
        0,
        0,
        1,
        1,
        0
    ],
    "dog": [
        0,
        0,
        1,
        0,
        1
    ],
    "bird": [
        0,
        1,
        0,
        0,
        0
    ],
    "horse": [
        1,
        0,
        0,
        0,
        0
    ],
    "snake": [
        0,
        1,
        0,
        0,
        0
    ],
    "prob_cat": [
        0.0000028932422337171306,
        3.038178716065753e-10,
        0.999976879069893,
        0.9998046130912787,
        0.000007710994236090884
    ],
    "prob_dog": [
        0.00001698134701966675,
        0.000057361347621279745,
        0.9999725147168369,
        0.000017608423499919837,
        0.9999473483590606
    ],
    "prob_bird": [
        1.0273657030762843e-7,
        0.9998969493283363,
        2.061596293323684e-8,
        0.000005218063433685078,
        0.00009392756109423784
    ],
    "prob_horse": [
        0.9999765195772874,
        1.1480034055759994e-8,
        1.7205732529738065e-7,
        0.00004734921618092327,
        0.000004443257653297175
    ],
    "prob_snake": [
        0.00000561530497076075,
        0.9999426335706552,
        2.026564423485328e-8,
        9.891596932479765e-8,
        0.00000329882753306212
    ]
}
MULTICLASS_DF = pd.DataFrame(MULTICLASS_RESPONSE).sort_values(by=['id']).reset_index(drop=True).round({
    'prob_cat': 5, 'prob_dog': 5, 'prob_bird': 5, 'prob_horse': 5, 'prob_snake': 5
})
MULTILABEL_DF = pd.DataFrame(MULTILABEL_RESPONSE).sort_values(by=['id']).reset_index(drop=True).round({
    'prob_cat': 5, 'prob_dog': 5, 'prob_bird': 5, 'prob_horse': 5, 'prob_snake': 5
})


def send_request(url, ids):
    r = requests.post(url,
                      headers={"accept": "application/json", "Content-Type": "application/json"},
                      data=json.dumps(
                          {
                              "table_name":
                                  [
                                      "example_data",
                                      "example_data",
                                      "example_data",
                                      "example_data",
                                      "example_data"
                                  ],
                              "id": ids
                          })
                      )
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        print(r.text)
        print("ERROR: Getting predictions failed.")
        sys.exit(1)
    return pd.DataFrame(r.json()).sort_values(by=['id']).reset_index(drop=True).round({
        'prob_cat': 5, 'prob_dog': 5, 'prob_bird': 5, 'prob_horse': 5, 'prob_snake': 5
    })


def check_response(df, expected):
    print("Checking predictions...")
    try:
        assert df.equals(expected)
    except AssertionError:
        print(df)
        print(expected)
        print("ERROR: Predictions were not as expected.")
        sys.exit(1)


print("=== STARTING MODEL DEPLOYMENT TESTS WITH EXAMPLE DATA ===")

print("Testing multiclass predictions...")
response = send_request("http://192.168.2.4:80/predict_multiclass_example",
                        ["22", "24", "19", "15", "18"])
check_response(response, MULTICLASS_DF)

print("Testing multilabel predictions...")
response = send_request("http://192.168.2.4:80/predict_multilabel_example",
                        ["03", "05", "12", "06", "08"])
check_response(response, MULTILABEL_DF)

print("=== MODEL DEPLOYMENT TESTS WITH EXAMPLE DATA COMPLETED SUCCESSFULLY ===")
