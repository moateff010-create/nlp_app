import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from sklearn.pipeline import make_pipeline
df = pd.read_csv(r"D:/datasets/twitter_training.csv",header = None)
df.columns = ["id","entity","sentiment","text"]
print(df.isnull().sum())
print(df.head())
df = df.dropna(subset=["text"])
x = df["text"]
y = df["sentiment"]
df_test = pd.read_csv(r"D:/datasets/twitter_validation.csv",header = None)
df_test.columns = ["id","entity","sentiment","text"]
x_test = df_test["text"]
y_test = df_test["sentiment"]

model = make_pipeline(TfidfVectorizer(max_features=5000,stop_words="english"),LogisticRegression(max_iter=1000,random_state=42))
model.fit(x,y)

y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

predictions_list = []
while True:
    user_input = input("enter your sentence : ")
    if user_input == "exit" :
        print("the app has ended")
        print(f"total predictions : {predictions_list}")
        break
    predictions = model.predict([user_input])[0]
    predictions_list.append((user_input,predictions))
    print(predictions)