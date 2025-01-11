import os
import sqlite3
import tempfile

from flask import Flask, render_template, request, redirect, url_for
from pytube import YouTube
import whisper

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Pour les fichiers temporaires éventuels
app.config['DB_PATH'] = 'database.db'

# Chargement d’un modèle Whisper
# Le modèle "tiny" ou "base" est plus rapide, "medium" ou "large" est plus précis.
model = whisper.load_model("base")


def init_db():
    """Fonction pour initialiser la connexion DB et la table si besoin."""
    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        # On s'assure que la table existe
        cursor.execute("""CREATE TABLE IF NOT EXISTS recipes (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            video_url TEXT NOT NULL,
                            title TEXT,
                            ingredients TEXT,
                            steps TEXT
                        );""")
        conn.commit()


@app.route('/')
def index():
    """Page d'accueil : liste des recettes déjà extraites."""
    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, video_url FROM recipes")
        data = cursor.fetchall()

    # data est une liste de tuples (id, title, video_url)
    return render_template('index.html', recipes=data)


@app.route('/add', methods=['GET', 'POST'])
def add_recipe():
    """
    Page /add : 
    - GET : Affiche un formulaire demandant l'URL de la vidéo
    - POST : Télécharge la vidéo, transcrit l'audio, extrait la recette et insère dans la DB
    """
    if request.method == 'POST':
        video_url = request.form.get('video_url')
        if not video_url:
            return "Veuillez fournir une URL de vidéo.", 400

        # 1. Téléchargement de la vidéo (audio) via pytube
        try:
            yt = YouTube(video_url)
            video_title = yt.title
            # On télécharge l'audio dans un dossier temporaire
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_stream = yt.streams.filter(only_audio=True).first()
                out_file = audio_stream.download(output_path=tmpdir)
                
                # 2. Transcription avec Whisper
                result = model.transcribe(out_file, language='fr')  # forcer FR si besoin
                full_text = result["text"]

                # 3. Extraction simplifiée des ingrédients / étapes
                #    (Ici on fait un traitement rudimentaire, à toi de l'améliorer)
                extracted_ingredients, extracted_steps = parse_recipe(full_text)

                # 4. Sauvegarde en base
                with sqlite3.connect(app.config['DB_PATH']) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""INSERT INTO recipes (video_url, title, ingredients, steps)
                                      VALUES (?, ?, ?, ?)""",
                                   (video_url, video_title, extracted_ingredients, extracted_steps))
                    conn.commit()

            return redirect(url_for('index'))

        except Exception as e:
            return f"Erreur lors du traitement : {e}", 500

    # Méthode GET : on affiche un simple formulaire
    return render_template('add_recipe.html')


@app.route('/recipe/<int:recipe_id>')
def view_recipe(recipe_id):
    """Affiche une recette en détail."""
    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, video_url, ingredients, steps FROM recipes WHERE id = ?", (recipe_id,))
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


def parse_recipe(transcript: str):
    """
    Fonction (très) rudimentaire pour séparer ingrédients et étapes 
    à partir d'un texte brut.
    
    Idéalement, tu utiliseras un vrai pipeline de NLP,
    mais voici un exemple minimaliste pour illustrer la logique.
    """
    # On cherche un mot clé "Ingrédients" et "Etapes" dans le texte par exemple
    # Attention, c'est très artisanal !
    # Dans la vraie vie, on ferait du NLP plus précis (SpaCy, GPT, Regex avancées, etc.)

    # Pour simplifier, on coupe en deux segments (si le mot "ingrédients" est trouvé).
    # On peut aussi chercher "étapes" ou "préparation".
    transcript_lower = transcript.lower()

    # Cherche la section "ingrédients" et la section "préparation" ou "étapes"
    # Pour un MVP, on fait un best-effort.
    ing_start = transcript_lower.find("ingrédient")
    prep_start = transcript_lower.find("étape")
    if prep_start == -1:
        prep_start = transcript_lower.find("préparation")

    if ing_start == -1 or prep_start == -1:
        # On n'a pas trouvé, on renvoie tout en "steps"
        return ("Aucun ingrédient détecté", transcript)
    
    # On isole la partie Ingrédients
    ingredients_part = transcript[ing_start:prep_start].strip()
    # La partie Étapes
    steps_part = transcript[prep_start:].strip()

    return (ingredients_part, steps_part)


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
