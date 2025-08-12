import pandas as pd
from transformers import pipeline

# ---------------------- Load Dataset ---------------------- #
def load_songs():
    df = pd.read_csv("playlist.csv")
    return df

songs_df = load_songs()

# ---------------------- Load LLM Sentiment Model ---------------------- #
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_model = load_model()

# ---------------------- Mood Detection Logic ---------------------- #
def detect_mood(text):
    text = text.lower()

    # Custom vibe-based mood detection (priority)
    if any(word in text for word in ["angry", "rage", "furious", "mad"]):
        return "😡 Angry"
    elif any(word in text for word in ["love", "romantic", "miss", "crush", "broke up"]):
        return "❤️ Romantic"
    elif any(word in text for word in ["gym", "phonk", "beast", "workout", "training"]):
        return "💪 Gym"
    elif any(word in text for word in ["sleep", "lofi", "calm", "relax", "chill"]):
        return "😴 Sleeping"
    elif any(word in text for word in ["party", "dance", "club", "weekend", "dj"]):
        return "🎉 Party"
    elif any(word in text for word in ["demotivated", "lost", "empty", "hopeless", "low", "failure"]):
        return "⚡ Motivational"

    # LLM-based sentiment detection
    result = sentiment_model(text)[0]
    label = result['label']

    if label == "POSITIVE":
        return "😄 Happy"
    elif label == "NEGATIVE":
        return "😢 Sad"
    else:
        return "😐 Neutral"

# ---------------------- Song Recommendation Logic ---------------------- #
def recommend_songs(mood):
    if "Happy" in mood:
        filtered = songs_df[songs_df["valence"] > 0.7]
    elif "Sad" in mood:
        filtered = songs_df[songs_df["valence"] < 0.3]
    elif "Neutral" in mood:
        filtered = songs_df[(songs_df["valence"] >= 0.3) & (songs_df["valence"] <= 0.7)]
    elif "Angry" in mood:
        filtered = songs_df[(songs_df["energy"] > 0.8) & (songs_df["valence"] < 0.4)]
    elif "Romantic" in mood:
        filtered = songs_df[(songs_df["valence"] > 0.6) & (songs_df["danceability"] > 0.5)]
    elif "Sleeping" in mood:
        filtered = songs_df[(songs_df["energy"] < 0.4) & (songs_df["valence"] >= 0.3)]
    elif "Gym" in mood:
        filtered = songs_df[(songs_df["energy"] > 0.8) & (songs_df["tempo"] > 120)]
    elif "Party" in mood:
        filtered = songs_df[(songs_df["danceability"] > 0.7) & (songs_df["energy"] > 0.7)]
    elif "Motivational" in mood:
        filtered = songs_df[(songs_df["valence"] > 0.6) & (songs_df["energy"] > 0.6)]
    else:
        filtered = songs_df

    recommended = filtered.sample(n=5) if len(filtered) >= 5 else filtered
    return recommended[["name", "artists"]]
