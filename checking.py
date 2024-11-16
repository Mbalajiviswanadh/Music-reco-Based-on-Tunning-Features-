import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import requests
from datetime import datetime

load_dotenv()

st.set_page_config(page_title="Music Recommendations from your Playlists", page_icon="🎵")

# Spotify API Configuration
CLIENT_ID = '965679437ffd487ca3e73347eb1b0a64'
CLIENT_SECRET = '5ceef646d96c43d0bd93f8c0a5e87d06'
API_BASE_URL = 'https://api.spotify.com/v1/'

# Initialize session state
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
    st.session_state.playlists = []
    st.session_state.songs = []
    st.session_state.music_df = None
    st.session_state.user_info = None

def get_client_credentials():
    """Get access token using client credentials flow"""
    auth_response = requests.post('https://accounts.spotify.com/api/token', {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
    })
    
    if auth_response.status_code == 200:
        auth_data = auth_response.json()
        return auth_data['access_token']
    return None

def fetch_user_info(user_id):
    """Fetch user profile information"""
    try:
        if not st.session_state.access_token:
            st.session_state.access_token = get_client_credentials()
            
        if st.session_state.access_token:
            headers = {'Authorization': f"Bearer {st.session_state.access_token}"}
            response = requests.get(f"{API_BASE_URL}users/{user_id}", headers=headers)
            
            if response.status_code == 200:
                st.session_state.user_info = response.json()
                return True
            else:
                st.error(f"Invalid user ID: {user_id}")
                # Clear previous session data
                st.session_state.user_info = None
                st.session_state.playlists = []
                st.session_state.songs = []
                st.session_state.music_df = None
                return False
    except Exception as e:
        st.error(f"Error fetching user info: {str(e)}")
        return False
    


def fetch_user_playlists(user_id):
    """Fetch user's public playlists"""
    try:
        headers = {'Authorization': f"Bearer {st.session_state.access_token}"}
        response = requests.get(f"{API_BASE_URL}users/{user_id}/playlists", headers=headers)
        
        if response.status_code == 200:
            playlists = response.json()['items']
            st.session_state.playlists = [
                {
                    'id': p['id'], 
                    'name': p['name'], 
                    'image': p['images'][0]['url'] if p['images'] else None
                } 
                for p in playlists
            ]
            return True
        else:
            st.error("Could not fetch playlists. Please check the user ID.")
            return False
    except Exception as e:
        st.error(f"Error fetching playlists: {str(e)}")
        return False
    
def get_audio_features(track_ids):
    """Get audio features for tracks using client credentials"""
    if not st.session_state.access_token:
        st.session_state.access_token = get_client_credentials()
        
    headers = {'Authorization': f"Bearer {st.session_state.access_token}"}
    audio_features = []
    
    # Process track IDs in chunks of 100 (Spotify API limit)
    for i in range(0, len(track_ids), 100):
        chunk = track_ids[i:i + 100]
        response = requests.get(f"{API_BASE_URL}audio-features?ids={','.join(chunk)}", headers=headers)
        if response.status_code == 200:
            audio_features.extend(response.json()['audio_features'])
    
    return audio_features

def get_tracks_info(track_ids):
    """Get track information using client credentials"""
    if not st.session_state.access_token:
        st.session_state.access_token = get_client_credentials()
        
    headers = {'Authorization': f"Bearer {st.session_state.access_token}"}
    tracks_info = []
    
    # Process track IDs in chunks of 50 (Spotify API limit)
    for i in range(0, len(track_ids), 50):
        chunk = track_ids[i:i + 50]
        response = requests.get(f"{API_BASE_URL}tracks?ids={','.join(chunk)}", headers=headers)
        if response.status_code == 200:
            tracks_info.extend(response.json()['tracks'])
    
    return tracks_info

def calculate_weighted_popularity(release_date):
    """Calculate popularity weight based on release date"""
    try:
        # Handle different date formats
        try:
            release_date = datetime.strptime(release_date, '%Y-%m-%d')
        except ValueError:
            release_date = datetime.strptime(release_date, '%Y')
        
        time_span = datetime.now() - release_date
        weight = 1 / (time_span.days + 1)
        return weight
    except:
        return 0.5  # Default weight if there's an error

def get_hybrid_recommendations(input_song_name, num_recommendations=5):
    """Get hybrid recommendations based on song features and popularity"""
    if st.session_state.music_df is None or input_song_name not in st.session_state.music_df['Track Name'].values:
        return []

    try:
        # Prepare features for content-based filtering
        features_to_use = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                          'speechiness', 'acousticness', 'instrumentalness', 
                          'liveness', 'valence', 'tempo']
        
        # Check if all required features are present
        missing_features = [f for f in features_to_use if f not in st.session_state.music_df.columns]
        if missing_features:
            st.warning(f"Missing audio features: {', '.join(missing_features)}")
            return []
        
        # Normalize features
        scaler = MinMaxScaler()
        music_features = st.session_state.music_df[features_to_use].values
        music_features_scaled = scaler.fit_transform(music_features)
        
        # Get input song index
        input_song_index = st.session_state.music_df[st.session_state.music_df['Track Name'] == input_song_name].index[0]
        
        # Calculate similarity scores using numpy
        input_features = music_features_scaled[input_song_index].reshape(1, -1)
        similarity_scores = np.dot(input_features, music_features_scaled.T) / (
            np.linalg.norm(input_features) * np.linalg.norm(music_features_scaled, axis=1)
        )
        
        # Get similar song indices
        similar_song_indices = similarity_scores.argsort()[0][::-1][1:num_recommendations + 5]
        
        # Get recommendations
        recommendations = st.session_state.music_df.iloc[similar_song_indices].copy()
        
        # Calculate weighted popularity
        recommendations['WeightedPopularity'] = recommendations.apply(
            lambda x: x['Popularity'] * calculate_weighted_popularity(x['Release Date']), axis=1
        )
        
        # Sort by weighted popularity and get top recommendations
        recommendations = recommendations.sort_values('WeightedPopularity', ascending=False)
        recommendations = recommendations.head(num_recommendations)
        
        return recommendations['Track ID'].tolist()
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []

def fetch_song_recommendations(track_id):
    """Fetch recommendations for a given track"""
    if not st.session_state.music_df is None:
        try:
            # Get the song name from track_id
            song_name = st.session_state.music_df[st.session_state.music_df['Track ID'] == track_id]['Track Name'].iloc[0]
            
            # Get recommendations using hybrid system
            recommended_track_ids = get_hybrid_recommendations(song_name)
            
            # Fetch full track information for the recommendations
            if recommended_track_ids:
                tracks_info = get_tracks_info(recommended_track_ids)
                return tracks_info
        except Exception as e:
            st.error(f"Error fetching recommendations: {str(e)}")
    return []

def fetch_playlist_songs(playlist_id):
    """Fetch songs from a playlist and process their data."""
    try:
        # Get access token from session state or fetch new one
        if not st.session_state.access_token:
            st.session_state.access_token = get_client_credentials()
        
        if st.session_state.access_token:
            headers = {'Authorization': f"Bearer {st.session_state.access_token}"}
            response = requests.get(f"{API_BASE_URL}playlists/{playlist_id}/tracks", headers=headers)
            
            if response.status_code == 200:
                tracks = response.json().get('items', [])
                
                # Check if playlist is empty
                if not tracks:
                    st.warning("No songs in playlist")
                    st.session_state.songs = []
                    st.session_state.music_df = None
                    return
                
                # Extract track IDs and basic info
                track_data = []
                track_ids = []
                
                for t in tracks:
                    if t['track'] and t['track']['id']:  # Check if track exists and has ID
                        track_ids.append(t['track']['id'])
                        track_data.append({
                            'id': t['track']['id'],
                            'name': t['track']['name'],
                            'artists': ', '.join([artist['name'] for artist in t['track']['artists']]),
                            'album_name': t['track']['album']['name'],
                            'release_date': t['track']['album']['release_date'],
                            'image': t['track']['album']['images'][0]['url'] if t['track']['album']['images'] else None
                        })
                
                if not track_data:
                    st.warning("No valid songs found in playlist")
                    st.session_state.songs = []
                    st.session_state.music_df = None
                    return
                
                # Store the simplified track data for display
                st.session_state.songs = track_data
                
                try:
                    # Get audio features and track info for recommendations
                    audio_features = get_audio_features(track_ids)
                    tracks_info = get_tracks_info(track_ids)
                    
                    # Create DataFrame for recommendations
                    df = pd.DataFrame(track_data)
                    df = df.rename(columns={
                        'id': 'Track ID',
                        'name': 'Track Name',
                        'artists': 'Artists',
                        'album_name': 'Album Name',
                        'release_date': 'Release Date',
                        'image': 'Image'
                    })
                    
                    # Create audio features DataFrame with proper error handling
                    audio_features_df = pd.DataFrame([af for af in audio_features if af is not None])
                    features_to_keep = ['danceability', 'energy', 'key', 'loudness', 'mode',
                                      'speechiness', 'acousticness', 'instrumentalness',
                                      'liveness', 'valence', 'tempo']
                    
                    # Check if all required features are present
                    if not all(feature in audio_features_df.columns for feature in features_to_keep):
                        missing_features = [f for f in features_to_keep if f not in audio_features_df.columns]
                        st.error(f"Missing audio features: {', '.join(missing_features)}")
                        st.session_state.music_df = None
                        return
                    
                    # Concatenate only if we have matching indices
                    if len(df) == len(audio_features_df):
                        df = pd.concat([df, audio_features_df[features_to_keep]], axis=1)
                    else:
                        st.error("Mismatch between track data and audio features")
                        st.session_state.music_df = None
                        return
                    
                    # Add popularity
                    popularity_data = {t['id']: t['popularity'] for t in tracks_info}
                    df['Popularity'] = df['Track ID'].map(popularity_data)
                    
                    # Store in session state
                    st.session_state.music_df = df
                
                except Exception as e:
                    st.error(f"Error processing playlist data: {str(e)}")
                    st.session_state.music_df = None
            else:
                st.error("Failed to fetch songs from playlist.")
                st.session_state.music_df = None
    except Exception as e:
        st.error(f"Error fetching playlist songs: {str(e)}")
        st.session_state.music_df = None


# Add this function to fetch playlist tracks with audio features
def fetch_playlist_tracks(playlist_id):
    """Fetch tracks from a playlist including audio features"""
    try:
        if not st.session_state.access_token:
            st.session_state.access_token = get_client_credentials()
            
        headers = {'Authorization': f"Bearer {st.session_state.access_token}"}
        response = requests.get(f"{API_BASE_URL}playlists/{playlist_id}/tracks", headers=headers)
        
        if response.status_code == 200:
            tracks = response.json()['items']
            
            if not tracks:
                st.warning("No songs in playlist")
                st.session_state.songs = []
                st.session_state.music_df = None
                return
            
            # Extract track data and IDs
            track_data = []
            track_ids = []
            
            for t in tracks:
                if t['track'] and t['track']['id']:
                    track_ids.append(t['track']['id'])
                    track_data.append({
                        'Track ID': t['track']['id'],
                        'Track Name': t['track']['name'],
                        'Artists': ', '.join([artist['name'] for artist in t['track']['artists']]),
                        'Album Name': t['track']['album']['name'],
                        'Release Date': t['track']['album']['release_date'],
                        'Image': t['track']['album']['images'][0]['url'] if t['track']['album']['images'] else None
                    })
            
            if not track_data:
                st.warning("No valid songs found in playlist")
                return
            
            # Create DataFrame
            df = pd.DataFrame(track_data)
            
            # Get audio features and track info
            audio_features = get_audio_features(track_ids)
            tracks_info = get_tracks_info(track_ids)
            
            # Add audio features to DataFrame
            audio_features_df = pd.DataFrame([af for af in audio_features if af is not None])
            features_to_keep = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                              'speechiness', 'acousticness', 'instrumentalness', 
                              'liveness', 'valence', 'tempo']
            
            if len(df) == len(audio_features_df):
                df = pd.concat([df, audio_features_df[features_to_keep]], axis=1)
            
            # Add popularity
            popularity_data = {t['id']: t['popularity'] for t in tracks_info}
            df['Popularity'] = df['Track ID'].map(popularity_data)
            
            # Store in session state
            st.session_state.songs = track_data
            st.session_state.music_df = df
            
        else:
            st.error("Failed to fetch playlist tracks")
            
    except Exception as e:
        st.error(f"Error fetching playlist tracks: {str(e)}")
        st.session_state.songs = []
        st.session_state.music_df = None

def main():
    st.title("Spotify Music Recommendation App")

    # Initialize session state if needed
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
        st.session_state.playlists = []
        st.session_state.songs = []
        st.session_state.music_df = None
        st.session_state.user_info = None

    # User ID input
    # user_id = st.text_input(
    #     "Enter Spotify User ID",
    #     help="You can find your Spotify User ID in your Spotify Account settings or profile URL"
    # )
    with st.form(key="user_id_form"):
        user_id = st.text_input(
            "Enter Spotify User ID",
            help="You can find your Spotify User ID in your Spotify Account settings or profile URL"
        )
        submit_button = st.form_submit_button("Enter")

    if user_id:
        # Try to fetch user info and playlists
        if fetch_user_info(user_id):
            fetch_user_playlists(user_id)
            
            # Display user info if available
            if st.session_state.user_info:
                col1, col2 = st.columns([1, 3])
                with col1:
                    if 'images' in st.session_state.user_info and st.session_state.user_info['images']:
                        st.markdown(
                            f"""
                            <style>
                                .responsive-profile-image {{
                                    display: inline-block;
                                    border-radius: 50%;
                                    overflow: hidden;
                                    border: 2px solid lightgreen;  
                                    width:35vw; 
                                    height: 35vw; 
                                    max-width: 100px; 
                                    max-height: 100px;
                                }}
                                .responsive-profile-image img {{
                                    width: 100%;
                                    height: 100%;
                                    object-fit: cover;
                                }}
                            </style>
                            <div class="responsive-profile-image">
                                <img src="{st.session_state.user_info['images'][0]['url']}" alt="Profile Image">
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with col2:
                    st.markdown(
                        f"""
                        <div>
                            Hello, <span style='color: lightgreen; font-size: 20px;'>{st.session_state.user_info.get('display_name', 'Unknown')}</span><br>
                            
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        
            st.markdown("___")
            
            # Only show playlist section if we have playlists
            if st.session_state.playlists:
                # Playlist dropdown
                playlist_names = [p['name'] for p in st.session_state.playlists]
                selected_playlist = st.selectbox("Choose a playlist", playlist_names)

                if selected_playlist:
                    selected_playlist_id = next(p['id'] for p in st.session_state.playlists if p['name'] == selected_playlist)
                    
                    with st.spinner("Loading playlist tracks..."):
                        fetch_playlist_songs(selected_playlist_id)

                    # Display playlist cover image
                    playlist_image = next(p['image'] for p in st.session_state.playlists if p['id'] == selected_playlist_id)
                    if playlist_image:
                        col1, col2 = st.columns([1, 3])  
                        with col1:
                            st.image(playlist_image, width=160)
                        with col2:
                            st.markdown(f"Selected Playlist: <span style='font-weight: bold; color: lightgreen;'>**{selected_playlist}**</span>", unsafe_allow_html=True)

                    if st.session_state.songs:
                        st.success(f"Loaded {len(st.session_state.songs)} tracks from the playlist!")
                
                        # Song dropdown for selected playlist
                        st.subheader("Get Recommendations")
                        song_names = [song.get('name', 'Unknown') for song in st.session_state.songs]
                        selected_song = st.selectbox("Choose a song", song_names)

                    
                        if selected_song:
                            # Find the selected song's index and details
                            selected_song_index = next((i for i, song in enumerate(st.session_state.songs) if song.get('name') == selected_song), None)
                            if selected_song_index is not None:
                                selected_song_id = st.session_state.songs[selected_song_index]['id']
                                song_image = st.session_state.songs[selected_song_index].get('image', None)

                                # Get the song features from the music dataframe
                                if st.session_state.music_df is not None and selected_song_id in st.session_state.music_df['Track ID'].values:
                                    song_features = st.session_state.music_df[
                                        st.session_state.music_df['Track ID'] == selected_song_id
                                    ].iloc[0]

                                    # Layout for image and details
                                    col1, col2 = st.columns([1, 3])  
                                    with col1:
                                        # Display song image
                                        if song_image:
                                            st.image(song_image, width=160)
                                    with col2:
                                        # Display song title and features
                                        st.markdown(
                                            f"""
                                            <div style='font-size: 1em;  margin-bottom: 10px;'>
                                                Selected Song: <span style='color: lightgreen; font-weight: bold;'>{selected_song}</span>
                                            </div>
                                            <div>Features of the Selected Song: </div>
                                            <div style='font-size: 0.8em; color: #888; padding: 10px 0;'>
                                                <span style='margin-right: 15px;'>🎭 Danceability: {song_features['danceability']:.2f}</span>
                                                <span style='margin-right: 15px;'>⚡ Energy: {song_features['energy']:.2f}</span>
                                                <span style='margin-right: 15px;'>🎵 Key: {song_features['key']}</span>
                                                <span style='margin-right: 15px;'>🔊 Loudness: {song_features['loudness']:.1f} dB</span>
                                                <br/>
                                                <span style='margin-right: 15px;'>🗣️ Speechiness: {song_features['speechiness']:.2f}</span>
                                                <span style='margin-right: 15px;'>🎸 Acousticness: {song_features['acousticness']:.2f}</span>
                                                <span style='margin-right: 15px;'>🎹 Instrumentalness: {song_features['instrumentalness']:.2f}</span>
                                                <br/>
                                                <span style='margin-right: 15px;'>🎪 Liveness: {song_features['liveness']:.2f}</span>
                                                <span style='margin-right: 15px;'>😊 Valence: {song_features['valence']:.2f}</span>
                                                <span style='margin-right: 15px;'>⏱️ Tempo: {song_features['tempo']:.0f} BPM</span>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                                else:
                                    st.write("Audio features not available for the selected song.")

                                # Fetch and display song recommendations
                                with st.spinner("Getting recommendations..."):
                                    recommendations = fetch_song_recommendations(selected_song_id)
                                
                                # ... (previous code remains the same until the recommendations display section)

                                if recommendations:
                                    st.subheader("Recommended Songs")
                                    for rec in recommendations:
                                        with st.expander(f"🎵 {rec.get('name', 'Unknown')} - {', '.join(artist['name'] for artist in rec['artists'])}"):
                                            # Spotify embed player
                                            embed_url = f"https://open.spotify.com/embed/track/{rec['id']}"
                                            st.markdown(
                                                f'<iframe style="border-radius:12px" src="{embed_url}" '
                                                'width="100%" height="152" frameBorder="0" allowfullscreen="" '
                                                'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" '
                                                'loading="lazy"></iframe>',
                                                unsafe_allow_html=True
                                            )
                                            
                                            # Get song features from music_df
                                            if st.session_state.music_df is not None and rec['id'] in st.session_state.music_df['Track ID'].values:
                                                song_features = st.session_state.music_df[
                                                    st.session_state.music_df['Track ID'] == rec['id']
                                                ].iloc[0]
                                                
                                                # Display audio features in a neat format
                                                st.markdown(
                                                    f"""
                                                    <div>Features of the song-</div>
                                                    <div style='font-size: 0.8em; color: #888; padding: 10px 0;'>
                                                        <span style='margin-right: 15px;'>🎭 Danceability: {song_features['danceability']:.2f}</span>
                                                        <span style='margin-right: 15px;'>⚡ Energy: {song_features['energy']:.2f}</span>
                                                        <span style='margin-right: 15px;'>🎵 Key: {song_features['key']}</span>
                                                        <span style='margin-right: 15px;'>🔊 Loudness: {song_features['loudness']:.1f} dB</span>
                                                        <br/>
                                                        <span style='margin-right: 15px;'>🗣️ Speechiness: {song_features['speechiness']:.2f}</span>
                                                        <span style='margin-right: 15px;'>🎸 Acousticness: {song_features['acousticness']:.2f}</span>
                                                        <span style='margin-right: 15px;'>🎹 Instrumentalness: {song_features['instrumentalness']:.2f}</span>
                                                        <br/>
                                                        <span style='margin-right: 15px;'>🎪 Liveness: {song_features['liveness']:.2f}</span>
                                                        <span style='margin-right: 15px;'>😊 Valence: {song_features['valence']:.2f}</span>
                                                        <span style='margin-right: 15px;'>⏱️ Tempo: {song_features['tempo']:.0f} BPM</span>
                                                    </div>
                                                    """,
                                                    unsafe_allow_html=True
                                                )

if __name__ == "__main__":
    main()