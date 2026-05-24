from flask import jsonify, request, render_template, session


def movie_route(app, tmdb_df, DEFAULT_POSTER):

    app.secret_key = "hamza_movie_recommender_123"

    # ================= MOVIE PAGE =================
    @app.route("/movie/<int:movie_id>")
    def movie(movie_id):

        movie_row = tmdb_df[tmdb_df["tmdb_id"] == movie_id]

        if movie_row.empty:
            return "Movie not found"

        movie = movie_row.iloc[0]

        # ================= RECENT =================
        if "recent_movies" not in session:
            session["recent_movies"] = []

        recent = session["recent_movies"]

        if movie_id in recent:
            recent.remove(movie_id)

        recent.insert(0, movie_id)

        session["recent_movies"] = recent[:10]

        # ================= GENRE UPDATE (RECENT = 1.5) =================
        if "genre_scores" not in session:
            session["genre_scores"] = {}

        genre_scores = session["genre_scores"]

        genres = str(movie["genres"]).split("|")

        for g in genres:
            g = g.strip()
            if g:
                genre_scores[g] = genre_scores.get(g, 0) + 1.5

        session["genre_scores"] = genre_scores

        return render_template("movie.html", movie=movie)


    # ================= ACTION ROUTE =================
    @app.route("/action", methods=["POST"])
    def action():

        movie_id = request.form.get("movie_id")
        action_type = request.form.get("type")

        if not movie_id:
            return "Movie ID missing"

        movie_id = int(movie_id)

        movie_row = tmdb_df[tmdb_df["tmdb_id"] == movie_id]

        if movie_row.empty:
            return "Movie not found"

        movie = movie_row.iloc[0]

        genres = str(movie["genres"]).split("|")

        # ================= GENRE STORAGE =================
        if "genre_scores" not in session:
            session["genre_scores"] = {}

        genre_scores = session["genre_scores"]

        # ================= FIXED WEIGHTS (YOUR IDEA) =================
        action_weights = {
            "click": 1.0,
            "watch_later": 1.2,
            "like": 2.0
        }

        weight = action_weights.get(action_type, 0)

        # ================= APPLY GENRE LEARNING =================
        for g in genres:
            g = g.strip()
            if not g:
                continue

            genre_scores[g] = genre_scores.get(g, 0) + weight

        session["genre_scores"] = genre_scores

        # ================= SAVE LISTS =================
        if action_type == "like":

            if "liked_movies" not in session:
                session["liked_movies"] = []

            liked = session["liked_movies"]

            if movie_id not in liked:
                liked.insert(0, movie_id)

            session["liked_movies"] = liked[:10]

            print("Genre Scores:", genre_scores)
            return "Liked Successfully"

        elif action_type == "watch_later":

            if "watch_later_movies" not in session:
                session["watch_later_movies"] = []

            watch = session["watch_later_movies"]

            if movie_id not in watch:
                watch.insert(0, movie_id)

            session["watch_later_movies"] = watch[:10]

            print("Genre Scores:", genre_scores)
            return "Added To Watch Later"

        elif action_type == "click":

            print("Genre Scores:", genre_scores)
            return "Click Saved"

        return "Done"