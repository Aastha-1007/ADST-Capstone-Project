# EVLoc AI — New EV Charging Station Location Recommendation

A complete Flask + SQLite + Random Forest academic project focused on **where to open a NEW EV charging station**.

## What the project does

1. Reads location and historical EV demand data.
2. Uses a Random Forest model to forecast EV charging demand.
3. Combines demand with EV population, traffic, existing station coverage, station gap and accessibility.
4. Produces a 0–100 location opportunity score.
5. Ranks Bengaluru areas and recommends where a new station should be considered.

## Pages

- Home
- About
- Login / Signup
- Dashboard
- AI Location Finder
- Location Rankings
- Opportunity Map
- Location Detail
- Analysis History
- Admin Dashboard

## Demo login

**Email:** admin@evlocai.com  
**Password:** admin123

## Windows setup

Open PowerShell inside the project folder:

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe app.py
```

Then open:

`http://127.0.0.1:5000`

### If PowerShell blocks Activate.ps1

You do **not** need to activate the environment. Use the commands above with `venv\Scripts\python.exe`.

## Regenerate the ML model

```powershell
venv\Scripts\python.exe train_model.py
```

The model and metrics are saved in `model/`.

## Project structure

```text
EVLoc_AI_Complete/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── run.bat
├── install.bat
├── dataset/
│   └── bengaluru_ev_location_data.csv
├── model/
│   ├── location_demand_model.pkl
│   └── metrics.json
├── database/
│   └── ev_location.db
├── static/
│   ├── css/style.css
│   └── js/app.js
└── templates/
    ├── base.html
    ├── home.html
    ├── about.html
    ├── login.html
    ├── signup.html
    ├── dashboard.html
    ├── find.html
    ├── rankings.html
    ├── map.html
    ├── location.html
    ├── history.html
    └── admin.html
```

## Dataset note

The included dataset contains **15,120 synthetic/demo historical records across 20 Bengaluru areas**. It is suitable for an academic prototype. For a real business system, replace it with verified charging transaction data, EV registration/population data, traffic data, road/network data, land availability, grid capacity and local regulatory information.

## ML model

The model uses:

- area
- nearby EV population
- traffic score
- existing stations
- distance to nearest station
- station gap score
- accessibility score
- hour
- day of week
- month
- weather

Target: `future_ev_demand`

The supplied model was saved with **scikit-learn 1.8.0**, and `requirements.txt` uses the same version to avoid pickle version warnings.
