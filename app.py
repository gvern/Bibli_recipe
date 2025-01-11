import os
import sqlite3
import tempfile

from flask import Flask, render_template, request, redirect, url_for
import yt_dlp
import whisper

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # For any temporary files if needed
app.config['DB_PATH'] = 'database.db'

# Load Whisper model ("base" by default; could be "tiny", "medium", or "large")
model = whisper.load_model("base")

def init_db():
    """Initialize the database and create the recipes table if it doesn't exist."""
    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_url TEXT NOT NULL,
                title TEXT,
                ingredients TEXT,
                steps TEXT
            );
        """)
        conn.commit()

@app.route('/')
def index():
    """List all recipes."""
    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, video_url FROM recipes")
        data = cursor.fetchall()
    # data is a list of tuples (id, title, video_url)
    return render_template('index.html', recipes=data)

@app.route('/add', methods=['GET', 'POST'])
def add_recipe():
    """
    /add route:
    - GET: show a form to input the YouTube video URL
    - POST: download the audio, transcribe, extract recipe parts, store in DB
    """
    if request.method == 'POST':
        video_url = request.form.get('video_url')
        if not video_url:
            return "Veuillez fournir une URL de vidéo.", 400

        try:
            # 1. Download audio with yt-dlp
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_path, video_title = download_audio_with_ytdlp(video_url, tmpdir)

                # 2. Transcribe with Whisper
                result = model.transcribe(audio_path, language='fr')  # force FR if needed
                full_text = result["text"]

                # 3. Naive extraction of ingredients / steps
                extracted_ingredients, extracted_steps = parse_recipe(full_text)

                # 4. Store in DB
                with sqlite3.connect(app.config['DB_PATH']) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO recipes (video_url, title, ingredients, steps)
                        VALUES (?, ?, ?, ?)
                    """, (video_url, video_title, extracted_ingredients, extracted_steps))
                    conn.commit()

            return redirect(url_for('index'))

        except Exception as e:
            return f"Erreur lors du traitement : {e}", 500

    # GET method: display simple form
    return render_template('add_recipe.html')

@app.route('/recipe/<int:recipe_id>')
def view_recipe(recipe_id):
    """Display a single recipe in detail."""
    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, video_url, ingredients, steps FROM recipes WHERE id = ?",
            (recipe_id,))
        row = cursor.fetchone()

    if not row:
        return "Recette introuvable.", 404

    recipe_data = {
        'id': row[0],
        'title': row[1],
        'video_url': row[2],
        'ingredients': row[3],
        'steps': row[4]
    }

    return render_template('recipe.html', recipe=recipe_data)

def download_audio_with_ytdlp(url, output_dir):
    """
    Downloads the best audio-only stream with yt-dlp,
    converts it to MP3 (or another format),
    and **forces 1 audio channel** to avoid multi-channel mismatch.
    Returns: (path_to_audio_file, video_title)
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': True,
        'noprogress': True,
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192'
            }
        ],
        # Force the output to have exactly 1 channel (mono)
        'postprocessor_args': [
            '-ac', '1'
        ]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_title = info.get('title', 'Titre inconnu')

    # The downloaded file likely ends with .mp3 now.
    # We can guess the final name by the title from info dict:
    sanitized_title = yt_dlp.utils.sanitize_filename(video_title)
    downloaded_file = os.path.join(output_dir, f"{sanitized_title}.mp3")

    return (downloaded_file, video_title)

def parse_recipe(transcript: str):
    """
    Very naive approach:
    - Looks for keywords "ingrédient(s)" and "étape(s)" or "préparation"
    - Splits the transcript into two parts
    """
    transcript_lower = transcript.lower()

    ing_start = transcript_lower.find("ingrédient")
    prep_start = transcript_lower.find("étape")
    if prep_start == -1:
        prep_start = transcript_lower.find("préparation")

    if ing_start == -1 or prep_start == -1:
        # If not found, everything goes to "steps"
        return ("Aucun ingrédient détecté", transcript)

    ingredients_part = transcript[ing_start:prep_start].strip()
    steps_part = transcript[prep_start:].strip()

    return (ingredients_part, steps_part)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
