import pandas as pd
from transformers import pipeline

# ---------------------- Load Dataset ---------------------- #
def load_songs():
    df = pd.read_csv("playlist.csv")
    for col in ["valence", "energy", "tempo", "danceability"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

songs_df = load_songs()

# ---------------------- Load LLM (zero-shot) ---------------------- #
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sentiment_model = load_model()

# ---------------------- Candidate moods ---------------------- #
CANDIDATE_MOODS = [
    "Happy", "Sad", "Neutral", "Angry", "Romantic",
    "Chill", "Gym", "Party", "Motivational",
    "Melancholic", "Confident", "Peaceful",
    "Excited", "Energetic", "Lonely", "Hopeful",
    "Nostalgic", "Relaxed", "Stressed", "Fearful", "Surprised"
]

EMOJI = {
    "Happy": "😄", "Sad": "😢", "Neutral": "😐", "Angry": "😡",
    "Romantic": "❤️", "Chill": "😌", "Gym": "💪", "Party": "🎉",
    "Motivational": "⚡", "Melancholic": "🌧", "Confident": "😎",
    "Peaceful": "🌅", "Excited": "🤩", "Energetic": "🔥",
    "Lonely": "😔", "Hopeful": "🌈", "Nostalgic": "📼",
    "Relaxed": "🛋️", "Stressed": "😣", "Fearful": "😱",
    "Surprised": "😲"
}

# ---------------------- Mood Detection ---------------------- #
def detect_mood(text: str) -> str:
    result = sentiment_model(text, CANDIDATE_MOODS)
    mood = result["labels"][0]
    return f"{EMOJI.get(mood, '')} {mood}"

# ---------------------- Recommendations per Mood ---------------------- #
def recommend_songs(mood: str):
    def between(s, lo, hi):
        return s.between(lo, hi, inclusive="both")

    df = songs_df.dropna(subset=["valence", "energy", "tempo", "danceability"])

    if "Happy" in mood:
        filtered = df[(df["valence"] > 0.75) & (df["energy"] > 0.60)]
    elif "Sad" in mood:
        filtered = df[(df["valence"] < 0.35) & (df["energy"] < 0.50)]
    elif "Neutral" in mood:
        filtered = df[between(df["valence"], 0.40, 0.60) & between(df["energy"], 0.40, 0.60)]
    elif "Angry" in mood:
        filtered = df[(df["energy"] > 0.85) & (df["valence"] < 0.40)]
    elif "Romantic" in mood:
        filtered = df[(df["valence"] > 0.65) & (df["danceability"] > 0.55) & (df["energy"] < 0.70)]
    elif "Chill" in mood:
        filtered = df[(df["energy"] < 0.45) & (df["valence"] > 0.40) & between(df["tempo"], 60, 100)]
    elif "Gym" in mood:
        filtered = df[(df["energy"] > 0.85) & (df["tempo"] > 120)]
    elif "Party" in mood:
        filtered = df[(df["danceability"] > 0.75) & (df["energy"] > 0.75) & (df["tempo"] > 110)]
    elif "Motivational" in mood:
        filtered = df[(df["valence"] > 0.65) & (df["energy"] > 0.65) & (df["tempo"] > 100)]
    elif "Melancholic" in mood:
        filtered = df[(df["valence"] < 0.45) & between(df["energy"], 0.30, 0.60) & between(df["tempo"], 60, 110)]
    elif "Confident" in mood:
        filtered = df[(df["energy"] > 0.70) & (df["valence"] > 0.50) & (df["danceability"] > 0.60) & (df["tempo"] > 100)]
    elif "Peaceful" in mood:
        filtered = df[(df["energy"] < 0.40) & between(df["valence"], 0.50, 0.80) & between(df["tempo"], 60, 90)]
    elif "Excited" in mood:
        filtered = df[(df["valence"] > 0.70) & (df["energy"] > 0.80) & (df["tempo"] > 110)]
    elif "Energetic" in mood:
        filtered = df[(df["energy"] > 0.90) & (df["tempo"] > 120)]
    elif "Lonely" in mood:
        filtered = df[(df["valence"] < 0.40) & (df["energy"] < 0.45)]
    elif "Hopeful" in mood:
        filtered = df[(df["valence"] > 0.55) & (df["energy"] > 0.50) & (df["tempo"] > 90)]
    elif "Nostalgic" in mood:
        filtered = df[(df["valence"] < 0.55) & between(df["tempo"], 70, 100)]
    elif "Relaxed" in mood:
        filtered = df[(df["energy"] < 0.50) & (df["tempo"] < 100)]
    elif "Stressed" in mood:
        filtered = df[(df["energy"] > 0.65) & (df["valence"] < 0.45)]
    elif "Fearful" in mood:
        filtered = df[(df["energy"] > 0.70) & (df["valence"] < 0.35)]
    elif "Surprised" in mood:
        filtered = df[between(df["valence"], 0.45, 0.70) & between(df["energy"], 0.50, 0.80)]
    else:
        filtered = df

    recommended = filtered.sample(n=5) if len(filtered) >= 5 else filtered
    return recommended[["name", "artists"]]
    
