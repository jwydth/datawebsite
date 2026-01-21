# Winter Shop — Data Science Group Project

## Student Information

- **Group Names:** DS_G3
- **Student Names:** Tran Viet Duc
- **Student IDs:** s4106117

- **Student Names:** To Minh Tuan
- **Student IDs:** s4055570

- **Student Names:** Huynh Huu Tri
- **Student IDs:** s4079860

---

## Project Overview

Winter Shop is a Flask-based web application for fashion item search, recommendation, and review analysis. It features:

- Advanced search with typo correction and suggestions
- Machine learning-based review recommendation (ensemble of LogisticRegression, SVM, LightGBM)
- TF-IDF weighted FastText vectors for text features
- Interactive web interface with dark/light mode

---

## Environment Setup

### Option A: Linux/MacOS

The encoded files (models.tar.gz.part-\*\*) need to be aggregated.

```sh
cat models.tar.gz.part-* | tar -xvzf –
```

### Option B — Windows PowerShell (no Linux needed)

Run the following **in PowerShell** from the folder that contains the `models.tar.gz.part-*` files.

#### Merge the parts into `models.tar.gz` (binary-safe)

`````powershell
@'
from pathlib import Path
import shutil

parts = sorted(Path(".").glob("models.tar.gz.part-*"))
if not parts:
    raise SystemExit("No parts found: models.tar.gz.part-*")

with open("models.tar.gz", "wb") as w:
    for p in parts:
        print("Adding", p.name)
        with open(p, "rb") as r:
            shutil.copyfileobj(r, w, 1024*1024)

print("✅ Created models.tar.gz")
'@ | python

````powershell


If you want to deploy the website manually, do the following instruction to set up the Anaconda environment. If you want to deploy the website using Docker, go to the next section.

### 1. Create and Activate Conda Environment, Please make sure the python environment is 3.12 (for gensim and matplot integration)

```sh
conda env create -n myenv python=3.12
conda activate myenv
`````

### 2. Download Libraries requirements

The app will attempt to download required NLTK resources automatically. If you encounter errors, run:

```sh
conda env update -f environment.yml -n myenv
```

### 3. Run the Application

```sh
python app.py
```

Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## Docker running

```sh
sudo docker build . -t datawebsite-img
sudo docker run --name datawebsite-con -p 5000:5000 datawebsite-img
```

Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## File Structure

- `app.py` — Main Flask application
- `models/` — Trained ML models and vectorizers
- `templates/` — HTML templates (Jinja2)
- `static/` — CSS and JavaScript files
- `data/` — CSV data and schema

---

## Model Information

- **Ensemble:** LogisticRegression, SVM, LightGBM
- **Features:** TF-IDF weighted FastText vectors
- **Training Data:** Processed review data from Milestone 1

Find out more about the process of building the models in the jupyter notebooks provided in the repository.

---

## Notes

- All Python and data science dependencies are managed via `requirements.yml` (Conda environment). This is due to the inconsistent download and building from the pip command
- Please use the conda install as instructions above.
- If you need to install additional packages, use `conda install <package>` or `pip install <package>` after activating the environment.
- For any issues with missing NLTK data, refer to step 2 above.

---

## Troubleshooting

- **Port in use:** If port 5000 is busy, change the port in `app.py` when running `app.run()`.
- **Model loading errors:** Ensure all files in `models/` are present and compatible with your Python version.
- **Search/ML errors:** Check that all required data files are in the `data/` directory.

---

## Contact

For questions or support, contact our group members. We are highly appreciate your support and guidance from Dr.Bao
