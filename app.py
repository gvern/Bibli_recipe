import os
import sqlite3
import tempfile
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for
import openai
from openai import OpenAI
import yt_dlp
import requests
import json


load_dotenv()
client = OpenAI()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # For any temporary files if needed
app.config['DB_PATH'] = 'database.db'

# Assurez-vous d'avoir défini OPENAI_API_KEY dans l'environnement
openai.api_key = os.getenv("OPENAI_API_KEY")

def init_db():
    """Initialize the database and create the table if it doesn't exist."""
    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_url TEXT NOT NULL,
                title TEXT,
                ingredients TEXT,
                steps TEXT,
                utensils TEXT,
                cook_time TEXT
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
    return render_template('index.html', recipes=data)

@app.route('/add', methods=['GET', 'POST'])
def add_recipe():
    """
    - GET: show a form to input the YouTube video URL
    - POST: download audio, call the Whisper endpoint manually,
      do basic extraction, then show a correction page before final save.
    """
    if request.method == 'POST':
        video_url = request.form.get('video_url')
        if not video_url:
            return "Veuillez fournir une URL de vidéo.", 400

        try:
            # 1. Download audio with yt-dlp
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_path, video_title = download_audio_with_ytdlp(video_url, tmpdir)

                # 2. Transcribe with OpenAI Whisper (via requête HTTP directe)
                raw_transcript = transcribe_with_openai_whisper(audio_path)

                # 3. Extraire infos (ingrédients, étapes, ustensiles, etc.)
                recipe_info = extract_recipe_info(raw_transcript)

                # 4. Affichage de la page de correction
                return render_template(
                    'correction.html',
                    video_url=video_url,
                    video_title=video_title,
                    raw_transcript=raw_transcript,
                    suggested_ingredients=recipe_info["ingredients"],
                    suggested_steps=recipe_info["steps"],
                    suggested_utensils=recipe_info["utensils"],
                    suggested_cook_time=recipe_info["cook_time"]
                )

        except Exception as e:
            return f"Erreur lors du traitement : {e}", 500

    # GET -> affiche le formulaire
    return render_template('add_recipe.html')

@app.route('/save_recipe', methods=['POST'])
def save_recipe():
    """
    After correction, the user hits "Save".
    We insert the final data into the DB.
    """
    video_url = request.form.get('video_url')
    video_title = request.form.get('video_title')
    ingredients = request.form.get('ingredients')
    steps = request.form.get('steps')
    utensils = request.form.get('utensils')
    cook_time = request.form.get('cook_time')

    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO recipes (video_url, title, ingredients, steps, utensils, cook_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (video_url, video_title, ingredients, steps, utensils, cook_time))
        conn.commit()

    return redirect(url_for('index'))

@app.route('/recipe/<int:recipe_id>')
def view_recipe(recipe_id):
    """Display a single recipe in detail."""
    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, video_url, ingredients, steps, utensils, cook_time
            FROM recipes WHERE id = ?
        """, (recipe_id,))
        row = cursor.fetchone()

    if not row:
        return "Recette introuvable.", 404

    recipe_data = {
        'id': row[0],
        'title': row[1],
        'video_url': row[2],
        'ingredients': row[3],
        'steps': row[4],
        'utensils': row[5],
        'cook_time': row[6]
    }
    return render_template('recipe.html', recipe=recipe_data)

# --------------------------
# HELPER FUNCTIONS
# --------------------------

def download_audio_with_ytdlp(url, output_dir):
    """
    Télécharge la vidéo YouTube en MP3 mono via yt-dlp,
    puis retourne (chemin_mp3, titre_video).
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
        'postprocessor_args': ['-ac', '1']  # force 1 audio channel
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_title = info.get('title', 'Titre inconnu')
    sanitized_title = yt_dlp.utils.sanitize_filename(video_title)
    audio_file = os.path.join(output_dir, f"{sanitized_title}.mp3")
    return (audio_file, video_title)

def transcribe_with_openai_whisper(audio_path):
    """
    Effectue un appel HTTP "manuel" à l'endpoint OpenAI 
    pour créer une transcription audio (Whisper).
    """
    # Endpoint officiel pour la transcription
    url = "https://api.openai.com/v1/audio/transcriptions"

    headers = {
        "Authorization": f"Bearer {openai.api_key}",
    }
    data = {
        "model": "whisper-1",
        # "language": "fr",  # Décommentez pour forcer le FR
    }

    # On envoie le fichier en multipart/form-data
    with open(audio_path, "rb") as f:
        files = {
            "file": (os.path.basename(audio_path), f, "audio/mpeg")
        }
        response = requests.post(url, headers=headers, data=data, files=files)

    if response.status_code != 200:
        raise Exception(f"OpenAI Whisper API error {response.status_code}: {response.text}")

    result = response.json()
    # 'result' contient un champ "text" si tout va bien
    return result.get("text", "")
def extract_recipe_info(raw_transcript):
    """
    Extrait ingrédients, ustensiles, temps de cuisson et étapes
    depuis la transcription brute.
    Utilise un prompt GPT plus précis pour mieux détecter.
    """
    try:
        prompt = f"""
Analyse le texte ci-dessous et extrais les informations suivantes pour une recette de cuisine :
- "ingredients": Une liste des ingrédients, chacun sous forme d'objet contenant "nom" (le nom ou descriptif) et "quantité" (le cas échéant, sinon "inconnu").
- "steps": Une liste des étapes successives de la préparation.
- "utensils": Une liste des ustensiles ou matériels de cuisine mentionnés.
- "cook_time": La durée totale de cuisson ou de repos (en minutes, sinon "inconnu").
- "prep_time": La durée de préparation (en minutes, sinon "inconnu").

Réponds uniquement en JSON strict. Exemple :

{
  "ingredients": [
    {"nom": "pommes de terre", "quantité": "1 kg"},
    {"nom": "gros sel", "quantité": "10 g"},
    {"nom": "lait entier", "quantité": "50 cl"},
    {"nom": "beurre", "quantité": "30 g"}
  ],
  "steps": ["Étape 1", "Étape 2", "Étape 3"],
  "utensils": ["couteau", "moulin à légumes", "casserole"],
  "cook_time": "25 minutes",
  "prep_time": "15 minutes"
}


Voici le texte :

\"\"\"{raw_transcript}\"\"\"
 
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant qui retourne uniquement du JSON strict. "
                        "Pas de texte explicatif ni de disclaimers."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0
        )


        data = json.loads(response.choices[0].message.content)

        # Récupérer les champs
        ingredients = data.get("ingredients", [])
        steps = data.get("steps", [])
        utensils = data.get("utensils", [])
        cook_time = data.get("cook_time", "inconnu")

        # Convertir en chaînes pour la DB
        if isinstance(ingredients, list):
            ingredients = "\n".join(ingredients)
        if isinstance(steps, list):
            steps = "\n".join(f"- {s}" for s in steps)
        if isinstance(utensils, list):
            utensils = ", ".join(utensils)

        return {
            "ingredients": ingredients,
            "steps": steps,
            "utensils": utensils,
            "cook_time": cook_time
        }

    except Exception as e:
        print("Erreur GPT extraction:", e)
        print(response.choices[0].message.content)
        # Fallback : on met tout dans steps
        return {
            "ingredients": "Ingrédients non détectés",
            "steps": raw_transcript,
            "utensils": "",
            "cook_time": "inconnu"
        }


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
