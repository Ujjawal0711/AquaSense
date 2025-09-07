# CMLRE Marine Data Prototype (Flask)

A 3-day SIH-ready backend prototype for a marine data platform.

## 1) Quick Start (Local)

```bash
# 1. Create venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install deps
pip install -r requirements.txt

# 3. Run
python app.py
# Server: http://127.0.0.1:5000/health
```

## 2) Test Endpoints

### Health
```
GET /health
```

### Upload CSV
```
POST /datasets/upload (multipart/form-data)
  file=@sample_data/ocean_sample.csv
  dataset_type=oceanography
  name=Ocean 2024
```

### Upload JSON
```
POST /datasets/upload (multipart/form-data)
  file=@sample_data/species_sample.json
  dataset_type=taxonomy
  name=Species Demo
```

### List datasets
```
GET /datasets
```

### Preview
```
GET /datasets/<id>/preview?limit=5
```

### Visualize
```
GET /datasets/<id>/visualize
# If columns ocean_temperature & fish_abundance exist -> scatter points returned.
```

### Species classify (stub)
```
POST /classify
Body: { "species": "Sardine" }
```

## 3) Cloud-Ready Notes

- Replace SQLite with Postgres by setting `DATABASE_URL` (see `.env.example`). Example:
  `postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME`
- For gunicorn (Render/Railway): Start command
  ```bash
  gunicorn app:app --bind 0.0.0.0:$PORT
  ```

## 4) File Tree

```
sih_cmlre_prototype/
├─ app.py
├─ database.py
├─ models.py
├─ requirements.txt
├─ .env.example
├─ sample_data/
│  ├─ ocean_sample.csv
│  └─ species_sample.json
└─ README.md
```

## 5) Postman (Optional)
Import endpoints manually, or use curl examples from above.
