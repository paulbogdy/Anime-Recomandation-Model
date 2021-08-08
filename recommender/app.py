from flask import Flask, render_template, url_for, request, redirect, flash
from markupsafe import escape
from utils import Recommender

app = Flask(__name__)
reco = Recommender()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/search", methods=['POST'])
def search():
    anime_name = request.form.get("search")
    anime_id = reco.anime_id(anime_name)
    if anime_id == -1:
        flash("No anime with that name found!")
        return redirect(url_for('home'))
    anime_name = reco.anime(anime_id)['Name']
    top15 = reco.top_x(anime_id)
    return render_template("recomandation.html", name=anime_name, len=15,
                           names=top15['Name'].values,
                           genres=top15['Genres'].values,
                           scores=top15['Score'].values,
                           episodes=top15['Episodes'].values,
                           descriptions=top15['Description'].values)


if __name__ == '__main__':
    app.secret_key = 'I love Hentai'
    app.run()
