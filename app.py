from flask import Flask, render_template, request
import sqlite3
from scipy.stats import poisson

app = Flask(__name__)

DB_NAME = "database.db"

# =====================================================
# DATABASE
# =====================================================

def get_db_connection():

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row

    return conn

# =====================================================
# HELPERS
# =====================================================

def normalize(value, max_value):

    value = value / max_value

    if value > 1:
        value = 1

    return value

def get_recent_matches(conn, team_name, limit=5):

    matches = conn.execute("""
    SELECT

    m.home_goals,
    m.away_goals,

    m.home_xg,
    m.away_xg,

    m.home_shots,
    m.away_shots,

    m.home_sot,
    m.away_sot,

    m.home_possession,
    m.away_possession,

    h.name,
    a.name

    FROM matches m

    JOIN teams h ON m.home_team_id = h.id
    JOIN teams a ON m.away_team_id = a.id

    WHERE h.name = ? OR a.name = ?

    ORDER BY m.date DESC
    LIMIT ?
    """, (team_name, team_name, limit)).fetchall()

    return matches

# =====================================================
# AI MODEL
# =====================================================

def get_team_strength(conn, team_name, is_home=True):

    matches = get_recent_matches(
        conn,
        team_name,
        5
    )

    if len(matches) < 3:
        return 1.1, 1.1

    attack_score = 0
    defense_weakness = 0

    total_weight = 0

    weight = 5

    for match in matches:

        hg, ag = match[0], match[1]

        hxg, axg = match[2], match[3]

        hs, ass = match[4], match[5]

        hsot, asot = match[6], match[7]

        hpos, apos = match[8], match[9]

        home_name = match[10]

        # TEAM IS HOME
        if home_name == team_name:

            goals_for = hg
            goals_against = ag

            xg_for = hxg
            xg_against = axg

            shots_for = hs
            shots_against = ass

            sot_for = hsot
            sot_against = asot

            pos_for = hpos

        # TEAM IS AWAY
        else:

            goals_for = ag
            goals_against = hg

            xg_for = axg
            xg_against = hxg

            shots_for = ass
            shots_against = hs

            sot_for = asot
            sot_against = hsot

            pos_for = apos

        # NORMALIZATION
        shots_score = normalize(shots_for, 25)

        sot_score = normalize(sot_for, 10)

        pos_score = normalize(pos_for, 100)

        # ATTACK
        attack = (
            goals_for * 0.35 +
            xg_for * 0.35 +
            shots_score * 0.10 +
            sot_score * 0.15 +
            pos_score * 0.05
        )

        # DEFENSE
        defense = (
            goals_against * 0.50 +
            xg_against * 0.30 +
            normalize(shots_against, 25) * 0.10 +
            normalize(sot_against, 10) * 0.10
        )

        attack_score += attack * weight

        defense_weakness += defense * weight

        total_weight += weight

        weight -= 1

    attack_score /= total_weight

    defense_weakness /= total_weight

    # HOME ADVANTAGE
    if is_home:
        attack_score *= 1.10

    # LIMITS
    attack_score = min(max(attack_score, 0.3), 4.0)

    defense_weakness = min(max(defense_weakness, 0.3), 4.0)

    return attack_score, defense_weakness

# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    conn = get_db_connection()

    matches = conn.execute("""
    SELECT

    m.date,

    h.name AS home_team,
    a.name AS away_team,

    m.home_goals,
    m.away_goals,

    m.home_xg,
    m.away_xg

    FROM matches m

    JOIN teams h ON m.home_team_id = h.id
    JOIN teams a ON m.away_team_id = a.id

    ORDER BY m.date DESC
    LIMIT 20
    """).fetchall()

    conn.close()

    return render_template(
        "index.html",
        matches=matches
    )

# =====================================================
# PREDICTIONS
# =====================================================

@app.route("/predictions", methods=["GET", "POST"])
def predictions():

    conn = get_db_connection()

    leagues = conn.execute("""
    SELECT name
    FROM leagues
    ORDER BY name
    """).fetchall()

    teams = []

    # LOAD FIRST LEAGUE AUTOMATICALLY
    if len(leagues) > 0:

        selected_league = leagues[0]["name"]

        teams = conn.execute("""
        SELECT teams.name

        FROM teams

        JOIN leagues
        ON teams.league_id = leagues.id

        WHERE leagues.name = ?

        ORDER BY teams.name
        """, (selected_league,)).fetchall()

    else:

        selected_league = None

    prediction = None

    # =================================================
    # POST
    # =================================================

    if request.method == "POST":

        selected_league = request.form.get("league")

        home_team = request.form.get("home_team")
        away_team = request.form.get("away_team")

        teams = conn.execute("""
        SELECT teams.name

        FROM teams

        JOIN leagues
        ON teams.league_id = leagues.id

        WHERE leagues.name = ?

        ORDER BY teams.name
        """, (selected_league,)).fetchall()

        if home_team and away_team and home_team != away_team:

            # =========================================
            # SAME ENGINE AS DESKTOP
            # =========================================

            home_attack, home_defense = get_team_strength(
                conn,
                home_team,
                True
            )

            away_attack, away_defense = get_team_strength(
                conn,
                away_team,
                False
            )

            home_lambda = (
                home_attack * away_defense
            ) / 1.5

            away_lambda = (
                away_attack * home_defense
            ) / 1.5

            home_lambda = min(max(home_lambda, 0.2), 4.5)

            away_lambda = min(max(away_lambda, 0.2), 4.5)

            # =========================================
            # POISSON
            # =========================================

            home_win = 0
            draw = 0
            away_win = 0

            over25 = 0
            btts = 0

            exact_scores = []

            for i in range(10):

                for j in range(10):

                    prob = (
                        poisson.pmf(i, home_lambda)
                        *
                        poisson.pmf(j, away_lambda)
                    )

                    exact_scores.append(
                        (f"{i}-{j}", prob)
                    )

                    if i > j:
                        home_win += prob

                    elif i == j:
                        draw += prob

                    else:
                        away_win += prob

                    if i + j > 2:
                        over25 += prob

                    if i > 0 and j > 0:
                        btts += prob

            exact_scores.sort(
                key=lambda x: x[1],
                reverse=True
            )

            prediction = {

                "home_team": home_team,
                "away_team": away_team,

                "home_win": round(home_win * 100, 2),
                "draw": round(draw * 100, 2),
                "away_win": round(away_win * 100, 2),

                "over25": round(over25 * 100, 2),
                "btts": round(btts * 100, 2),

                "home_xg": round(home_lambda, 2),
                "away_xg": round(away_lambda, 2),

                "scores": exact_scores[:5]
            }

    conn.close()

    return render_template(
        "predictions.html",
        leagues=leagues,
        teams=teams,
        prediction=prediction,
        selected_league=selected_league
    )

# =====================================================
# START
# =====================================================

if __name__ == "__main__":

    app.run(debug=True)