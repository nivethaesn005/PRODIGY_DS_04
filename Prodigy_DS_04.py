import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

train_df = pd.read_csv(r"D:\Prodigy Infotech\Prodigy_Task-04\twitter_training.csv")
val_df = pd.read_csv(r"D:\Prodigy Infotech\Prodigy_Task-04\twitter_validation.csv")
columns = ['ID', 'Entity', 'Sentiment', 'Text']
train_df.columns = columns
val_df.columns = columns
df = pd.concat([train_df, val_df], ignore_index=True)

df['Text'] = df['Text'].astype(str).str.lower().str.strip()
df['Sentiment'] = df['Sentiment'].str.capitalize()

df['Clean_Text'] = df['Text'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

print("\n📊 Tweet Count by Sentiment:")
print(df['Sentiment'].value_counts())

print("\n🔥 Top 5 Entities by Tweet Volume:")
print(df['Entity'].value_counts().head(5))

#Line Plot
sentiment_trend = df.groupby('Sentiment').size().reset_index(name='Count')
plt.figure(figsize=(8, 5))
sns.lineplot(data=sentiment_trend, x='Sentiment', y='Count', marker='o', linewidth=2)
plt.title("Sentiment Frequency Across Dataset")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#Countplot
top_entities = df['Entity'].value_counts().nlargest(6).index
plt.figure(figsize=(10, 6))
sns.countplot(data=df[df['Entity'].isin(top_entities)], x='Entity', hue='Sentiment', palette='Set2')
plt.title("Tweet Volume by Entity and Sentiment")
plt.xlabel("Entity")
plt.ylabel("Tweet Count")
plt.tight_layout()
plt.show()

#Terminal Print
def top_words_by_sentiment(sentiment, num=10):
    text = " ".join(df[df['Sentiment'] == sentiment]['Clean_Text'])
    words = [word for word in text.split() if word not in stop_words and len(word) > 3]
    most_common = Counter(words).most_common(num)
    print(f"\n🔍 Top {num} Words in {sentiment} Tweets:")
    for word, count in most_common:
        print(f"{word}: {count}")

top_words_by_sentiment("Positive")
top_words_by_sentiment("Negative")
top_words_by_sentiment("Neutral")