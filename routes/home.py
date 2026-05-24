from flask import jsonify, request, render_template, session


def home_route(app, tmdb_df, DEFAULT_POSTER):

    @app.route("/")
    def home():
        return render_template("index.html")


    # ================= MOVIES API =================
    @app.route("/api/movies")
    def get_movies():

        page = int(request.args.get("page", 1))
        limit = 30

        start = (page - 1) * limit
        end = start + limit

        movies_slice = tmdb_df.iloc[start:end]

        movie_list = []

        for _, movie in movies_slice.iterrows():

            poster_url = movie["poster_url"]

            if poster_url == "" or str(poster_url) == "nan":
                poster_url = DEFAULT_POSTER

            overview = movie["overview"]

            if overview == "" or str(overview) == "nan":
                overview = "No overview available."

            movie_list.append({
                "MovieID": movie["tmdb_id"],
                "title": movie["title"],
                "genres": movie["genres"],
                "poster_url": poster_url,
                "overview": overview
            })

        return jsonify(movie_list)


    # ================= GENRE CLICK LEARNING =================
    @app.route("/movie-click", methods=["POST"])
    def movie_click():

        movie_id = request.form.get("movie_id")
        action_type = request.form.get("type", "click")

        if not movie_id:
            return "Movie ID Missing"

        movie_id = int(movie_id)

        movie_row = tmdb_df[tmdb_df["tmdb_id"] == movie_id]

        if movie_row.empty:
            return "Movie Not Found"

        movie = movie_row.iloc[0]

        genres = str(movie["genres"]).split("|")

        # ================= INIT =================
        if "genre_scores" not in session:
            session["genre_scores"] = {}

        genre_scores = session["genre_scores"]

        # ================= CORRECT WEIGHTS =================
        weights = {
            "click": 1.0,
            "watch_later": 1.2,
            "like": 2.0,
            "recent": 1.5
        }

        weight = weights.get(action_type, 1.0)

        # ================= UPDATE GENRES =================
        for g in genres:

            g = g.strip()

            if not g:
                continue

            genre_scores[g] = genre_scores.get(g, 0) + weight

        session["genre_scores"] = genre_scores

        print("Updated Genre Scores:", genre_scores)

        return "Genre Updated Successfully"