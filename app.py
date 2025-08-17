import streamlit as st
from logic import detect_mood, recommend_songs
from googleapiclient.discovery import build

# YOUTUBE API SETUP
API_KEY = API_KEY = st.secrets["API_KEY"]
youtube = build("youtube", "v3", developerKey=API_KEY)

def get_youtube_video_id(query):
    try:
        avoid = ["cover", "live", "shorts", "#shorts"] 
        official_keywords = ["official", "lyrics", "lyrical"]
        aesthetic_keywords = ["lofi", "slowed", "reverb"]

        search_query = f"{query} official OR lyrics OR lyrical OR slowed OR reverb OR lofi"
        request = youtube.search().list(
            q=search_query,
            part="snippet",
            maxResults=10,
            type="video",
            videoEmbeddable="true"
        )
        response = request.execute()

        if "items" in response:
            query_lower = query.lower()

            #Official match
            for item in response["items"]:
                title = item["snippet"]["title"].lower()
                channel = item["snippet"]["channelTitle"].lower()
                if (any(word in title for word in official_keywords) or channel in query_lower) \
                        and not any(bad in title for bad in avoid):
                    return item["id"]["videoId"]

            #Aesthetic match
            for item in response["items"]:
                title = item["snippet"]["title"].lower()
                if any(word in title for word in aesthetic_keywords) \
                        and not any(bad in title for bad in avoid):
                    return item["id"]["videoId"]

            #Any clean video
            for item in response["items"]:
                title = item["snippet"]["title"].lower()
                if not any(bad in title for bad in avoid):
                    return item["id"]["videoId"]

    except Exception as e:
        st.error(f"Error fetching YouTube video: {e}")
    return None

# Streamlit Page Config
st.set_page_config(page_title="Mood2Music 🎧", page_icon="🎶", layout="centered")

#CSS Styling 
st.markdown("""
    <style>
    body { background-color: black; }
    h1, h4 { color: white; text-shadow: 0 0 6px #00f7ff, 0 0 12px #00f7ff, 0 0 24px #00f7ff; }
    .song-card {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 14px 0;
        backdrop-filter: blur(6px);
        border: 2px solid #00f7ff;
        box-shadow: 0 0 10px #00f7ff, 0 0 20px #00f7ff, 0 0 30px #00f7ff;
        position: relative;
        z-index: 3;
    }
    div.stButton > button {
        background-color: #00f7ff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        box-shadow: 0 0 5px #00f7ff, 0 0 15px #00f7ff;
        transition: all 0.18s ease-in-out;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 10px #00f7ff, 0 0 25px #00f7ff;
    }
    .stTextInput>div>div>input { background: rgba(255,255,255,0.02); color: white; }
    .stMarkdown p, .stText, .stHeader { text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6); }
    </style>
""", unsafe_allow_html=True)

#Title 
st.markdown("<h1 style='text-align: center;'>Mood2Music 🎶</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Your AI Mood Detector & Song Recommender</h4>", unsafe_allow_html=True)

#User Input
st.markdown("<h4 style='color: white;'>📝 Describe Your Current Mood</h4>", unsafe_allow_html=True)
user_input = st.text_input("", placeholder="e.g., I am ready to hit Gym today")

# Recommendation Output
if st.button("🎧 Recommend Songs"):
    if user_input.strip() == "":
        st.warning("Please enter something first.")
    else:
        try:
            detected_mood = detect_mood(user_input)
        except Exception as e:
            st.error(f"Error in mood detection: {e}")
            raise

        st.success(f"Detected Mood: {detected_mood}")

        # Extract mood without emoji
        mood_label = detected_mood.split(" ", 1)[1] if " " in detected_mood else detected_mood

        try:
            recommendations = recommend_songs(mood_label)
        except Exception as e:
            st.error(f"Error while recommending songs: {e}")
            raise

        if recommendations is None or recommendations.empty:
            st.info("No matching songs found for this mood. Try a different description.")
        else:
            st.markdown("**🎵 Top 5 Song Recommendations:**")
            for i, row in enumerate(recommendations.itertuples(), 1):
                song_name = row.name
                artist = row.artists
                query = f"{song_name} {artist}"

                video_id = get_youtube_video_id(query)
                if video_id:
                    embed_url = f"https://www.youtube.com/embed/{video_id}"
                    st.markdown(f"""
                        <div class="song-card">
                            <b>{i}. {song_name}</b> – {artist} <br><br>
                            <iframe width="100%" height="200" src="{embed_url}" 
                                frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                                allowfullscreen></iframe>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="song-card">
                            <b>{i}. {song_name}</b> – {artist} <br>
                            ❌ Preview not available
                        </div>
                    """, unsafe_allow_html=True)
