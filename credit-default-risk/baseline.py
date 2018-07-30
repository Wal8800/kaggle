import process_data
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


train_data = process_data.read_train()
train_label = train_data["TARGET"]
train_data.drop("TARGET", axis=1, inplace=True)

feature_names = [
    "EXT_SOURCE_2",
    "EXT_SOURCE_1",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "REGION_RATING_CLIENT_W_CITY",
    "CODE_GENDER"
]

train_data, cate_feats = process_data.process_data(train_data, feature_names, always_label_encode=True,
                                                   fill_null_columns=True)

scores = cross_val_score(LogisticRegression(C=0.001), train_data, train_label, cv=5, scoring='roc_auc')
print(scores)

scores = cross_val_score(RandomForestClassifier(), train_data, train_label, cv=5, scoring='roc_auc')
print(scores)

# [0.60062005 0.59834803 0.60353034 0.59639479 0.60282475]
# [0.65144001 0.64819679 0.64981109 0.64842029 0.6493538 ]


