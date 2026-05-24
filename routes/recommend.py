import numpy as np
from flask import session, render_template


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def recommend_route(app, request, render_template, tmdb_df, DEFAULT_POSTER):

    @app.route("/recommend")
    def recommend():

        # ================= LOAD RBM =================
        W = np.load("rbm_weights.npy")
        vb = np.load("rbm_visible_bias.npy")
        hb = np.load("rbm_hidden_bias.npy")

        n_movies = W.shape[0]

        # ================= USER DATA =================
        liked = session.get("liked_movies", [])
        watch = session.get("watch_later_movies", [])
        recent = session.get("recent_movies", [])

        if len(liked + watch + recent) == 0:
            return render_template("recommendations.html", movies=[])

        seen_movies = set(liked + watch + recent)

        # ================= GENRE PROFILE =================
        genre_scores = session.get("genre_scores", {})

        # DEBUG OUTPUT (clean)
        print("\n===== USER GENRE PROFILE =====")
        for k, v in sorted(genre_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{k}: {v}")
        print("================================\n")

        

        # ================= MAPPING =================
        movie_ids = list(tmdb_df["tmdb_id"].values)
        id_to_index = {movie_ids[i]: i for i in range(len(movie_ids))}

        # ================= USER VECTOR =================
        user_vector = np.zeros(n_movies)

        # Recent (strong + time decay)
        for i, m in enumerate(recent):
            if m in id_to_index:
                idx = id_to_index[m]
                user_vector[idx] += max(1.5 - i * 0.1, 0.3)

        # Liked (strongest signal)
        for i, m in enumerate(liked):
            if m in id_to_index:
                idx = id_to_index[m]
                user_vector[idx] += max(2.0 - i * 0.15, 0.4)

        # Watch later (medium signal)
        for i, m in enumerate(watch):
            if m in id_to_index:
                idx = id_to_index[m]
                user_vector[idx] += max(1.2 - i * 0.1, 0.2)

        # Seen boost
        for m in seen_movies:
            if m in id_to_index:
                user_vector[id_to_index[m]] += 0.5

        # ================= NORMALIZE =================
        max_val = np.max(user_vector)
        if max_val > 0:
            user_vector = user_vector / (max_val + 1e-8)

        # ================= RBM =================
        hidden = sigmoid(np.dot(user_vector, W) + hb)
        reconstructed = sigmoid(np.dot(hidden, W.T) + vb)

        reconstructed += np.random.normal(0, 0.01, reconstructed.shape)

        # ================= TOP-K =================
        top_k = np.argsort(reconstructed)[::-1][:250]
        np.random.shuffle(top_k)

        # ================= BUILD RECOMMENDATIONS =================
        recommendations = []
        used_genres = set()

        for i in top_k:

            if i >= len(tmdb_df):
                continue

            movie_id = movie_ids[i]

            if movie_id in seen_movies:
                continue

            score = float(reconstructed[i])

            if score < 0.25:
                continue

            movie = tmdb_df.iloc[i]

            genres = str(movie.get("genres", "")).split("|")
            genres = [g.strip() for g in genres if g.strip()]

            

            # ================= GENRE BOOST =================
            genre_boost = 0
            for g in genres:
                genre_boost += genre_scores.get(g, 0) * 0.05

            # ================= DIVERSITY BOOST =================
            diversity_bonus = len([g for g in genres if g not in used_genres]) * 0.05

            final_score = score + genre_boost + diversity_bonus

            recommendations.append({
                "title": movie["title"],
                "genres": movie.get("genres", ""),
                "overview": movie.get("overview", ""),
                "poster_url": movie.get("poster_url") or DEFAULT_POSTER,
                "score": round(final_score, 4)
            })

            for g in genres:
                used_genres.add(g)

            if len(recommendations) >= 10:
                break

        # ================= FINAL SORT =================
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        return render_template(
            "recommendations.html",
            movies=recommendations,
            genre_scores=genre_scores
        )