# sentiment_analysis.py
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
df = pd.read_csv(r"C:\Users\ROHAN\Downloads\CODTECH\CODTECH_Task4\Reviews.csv")
# Check the column name and rename if necessary
if 'Text' not in df.columns:
    print("Error: Column 'Text' not found in the dataset.")
    print("Available columns:", df.columns)
    exit()
# Clean missing values
df = df[['Text']].dropna()
# Apply TextBlob for sentiment polarity
df['Polarity'] = df['Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
# Classify sentiment
def get_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'
df['Sentiment'] = df['Polarity'].apply(get_sentiment)
# Print sample results
print(df[['Text', 'Polarity', 'Sentiment']].head())
# Visualize the sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Sentiment', order=['Positive', 'Negative', 'Neutral'], palette='Set2')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()
