from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os, sqlite3, hashlib, joblib, pandas as pd, numpy as np
from functools import wraps
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, 'database', 'ev_location.db')
MODEL_PATH = os.path.join(BASE, 'model', 'location_demand_model.pkl')
DATA_PATH = os.path.join(BASE, 'dataset', 'bengaluru_ev_location_data.csv')

app = Flask(__name__)
app.secret_key = os.environ.get('EVLOC_SECRET', 'evloc-ai-demo-secret-change-me')
app.config['TEMPLATES_AUTO_RELOAD'] = True

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# The location-level fields are taken from the dataset so the app stays consistent with the ML data.
AREAS = []
for row in df.drop_duplicates('area').itertuples(index=False):
    AREAS.append({
        'area': row.area,
        'lat': float(row.latitude),
        'lon': float(row.longitude),
        'ev': int(row.nearby_ev_population),
        'traffic': float(row.traffic_score),
        'existing': int(row.existing_stations),
        'distance': float(row.distance_to_nearest_station_km),
        'gap': float(row.station_gap_score),
        'access': float(row.accessibility_score),
    })

FEATURES = [
    'area', 'nearby_ev_population', 'traffic_score', 'existing_stations',
    'distance_to_nearest_station_km', 'station_gap_score',
    'accessibility_score', 'hour', 'day_of_week', 'month', 'weather'
]

# The model was trained with pandas categorical/string columns. Keep the input shape consistent.
def predict_demand(location, hour=18, day_of_week=4, month=10, weather='Clear'):
    x = pd.DataFrame([{
        'area': location['area'],
        'nearby_ev_population': location['ev'],
        'traffic_score': location['traffic'],
        'existing_stations': location['existing'],
        'distance_to_nearest_station_km': location['distance'],
        'station_gap_score': location['gap'],
        'accessibility_score': location['access'],
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'weather': weather,
    }])[FEATURES]
    return max(0.0, float(model.predict(x)[0]))


def location_score(location, demand=None):
    if demand is None:
        demand = predict_demand(location)
    demand_component = min(100, max(0, demand * 1.9))
    ev_component = min(100, location['ev'] / 10)
    gap_component = min(100, location['gap'])
    traffic_component = min(100, location['traffic'])
    access_component = min(100, location['access'])
    competition_component = max(0, 100 - location['existing'] * 4)
    score = (
        0.32 * demand_component +
        0.20 * ev_component +
        0.18 * gap_component +
        0.12 * traffic_component +
        0.10 * access_component +
        0.08 * competition_component
    )
    return round(min(100, max(0, score)), 1)


def recommendation(score):
    if score >= 80:
        return 'Excellent opportunity', 'OPEN NEW STATION', 'strong'
    if score >= 70:
        return 'High-potential location', 'OPEN NEW STATION', 'good'
    if score >= 60:
        return 'Promising — validate locally', 'SHORTLIST', 'watch'
    return 'Lower priority for now', 'MONITOR', 'low'


def rankings():
    now = datetime.now()
    out = []
    for loc in AREAS:
        # Average three useful operating periods for a stable location-level forecast.
        forecasts = [
            predict_demand(loc, 8, 1, now.month, 'Clear'),
            predict_demand(loc, 14, 3, now.month, 'Cloudy'),
            predict_demand(loc, 18, 4, now.month, 'Clear'),
        ]
        demand = round(float(np.mean(forecasts)), 1)
        score = location_score(loc, demand)
        label, action, tone = recommendation(score)
        out.append({**loc, 'demand': demand, 'score': score, 'label': label, 'action': action, 'tone': tone})
    return sorted(out, key=lambda x: x['score'], reverse=True)


def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            area TEXT,
            predicted_demand REAL,
            location_score REAL,
            recommendation TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    admin = conn.execute('SELECT id FROM users WHERE email=?', ('admin@evlocai.com',)).fetchone()
    if not admin:
        pwd = hashlib.sha256('admin123'.encode()).hexdigest()
        conn.execute('INSERT INTO users(name,email,password_hash,role) VALUES(?,?,?,?)',
                     ('EVLoc Admin', 'admin@evlocai.com', pwd, 'admin'))
    conn.commit()
    conn.close()


def auth_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('uid'):
            flash('Please log in to continue.', 'info')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapped


def admin_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if session.get('role') != 'admin':
            flash('Admin access required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return wrapped


@app.context_processor
def globals_for_templates():
    return {'year': datetime.now().year}


@app.route('/')
def home():
    top = rankings()[0]
    return render_template('home.html', top=top, area_count=len(AREAS), record_count=len(df))


@app.route('/about')
def about():
    return render_template('about.html', area_count=len(AREAS), record_count=len(df))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if len(name) < 2 or '@' not in email or len(password) < 6:
            flash('Please enter a valid name, email and password of at least 6 characters.', 'error')
            return render_template('signup.html')
        conn = db()
        try:
            ph = hashlib.sha256(password.encode()).hexdigest()
            conn.execute('INSERT INTO users(name,email,password_hash) VALUES(?,?,?)', (name, email, ph))
            conn.commit()
            flash('Account created successfully. Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('That email is already registered.', 'error')
        finally:
            conn.close()
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        ph = hashlib.sha256(password.encode()).hexdigest()
        conn = db()
        user = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
        conn.close()
        if user and user['password_hash'] == ph:
            session.update(uid=user['id'], name=user['name'], role=user['role'])
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


@app.route('/dashboard')
@auth_required
def dashboard():
    rows = rankings()
    conn = db()
    analyses = conn.execute('SELECT * FROM analyses WHERE user_id=? ORDER BY id DESC LIMIT 5', (session['uid'],)).fetchall()
    analysis_count = conn.execute('SELECT COUNT(*) c FROM analyses WHERE user_id=?', (session['uid'],)).fetchone()['c']
    conn.close()
    return render_template('dashboard.html', rows=rows, best=rows[0], recent=analyses, analysis_count=analysis_count)


@app.route('/find-location', methods=['GET', 'POST'])
@auth_required
def find_location():
    rows = rankings()
    result = None
    if request.method == 'POST':
        selected = request.form.get('area', 'ALL')
        result = rows[0] if selected == 'ALL' else next((r for r in rows if r['area'] == selected), rows[0])
        conn = db()
        conn.execute('INSERT INTO analyses(user_id,area,predicted_demand,location_score,recommendation) VALUES(?,?,?,?,?)',
                     (session['uid'], result['area'], result['demand'], result['score'], result['action']))
        conn.commit(); conn.close()
    return render_template('find.html', rows=rows, areas=[r['area'] for r in rows], result=result)


@app.route('/rankings')
@auth_required
def rankings_page():
    return render_template('rankings.html', rows=rankings())


@app.route('/map')
@auth_required
def map_page():
    return render_template('map.html', rows=rankings())


@app.route('/location/<path:name>')
@auth_required
def location(name):
    row = next((r for r in rankings() if r['area'] == name), None)
    if not row:
        flash('Location not found.', 'error')
        return redirect(url_for('rankings_page'))
    return render_template('location.html', r=row)


@app.route('/history')
@auth_required
def history():
    conn = db()
    rows = conn.execute('SELECT * FROM analyses WHERE user_id=? ORDER BY id DESC', (session['uid'],)).fetchall()
    conn.close()
    return render_template('history.html', rows=rows)


@app.route('/admin')
@admin_required
def admin_page():
    conn = db()
    users = conn.execute('SELECT id,name,email,role,created_at FROM users ORDER BY id DESC').fetchall()
    analyses = conn.execute('SELECT COUNT(*) c FROM analyses').fetchone()['c']
    conn.close()
    return render_template('admin.html', users=users, analyses=analyses, rows=rankings())


@app.route('/api/rankings')
@auth_required
def api_rankings():
    return jsonify(rankings())


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
