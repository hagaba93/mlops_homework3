# Lab 4: Model Training and Serving with Airflow + FastAPI

In this lab you will build an **end-to-end ML pipeline** using Apache Airflow and serve the trained model with FastAPI.  

The pipeline includes:  
1. **Generate Data** – downloads the Iris dataset and saves it as a CSV.  
2. **Train Model** – trains a Logistic Regression classifier.  
3. **Pipeline** – runs both steps end-to-end.  
4. **Serve Model** – starts a FastAPI app for inference.  

---

## 📂 Project Structure

```
lab4_model_training/
├── dags/                        # Airflow DAGs
│   ├── ml_pipeline_dag.py       # full pipeline: generate + train
│   ├── generate_data_dag.py     # generate dataset only
│   └── train_model_dag.py       # train model only
├── src/
│   ├── ml_pipeline/             # training pipeline
│   │   ├── data.py
│   │   └── model.py
│   └── app/                     # serving app
│       └── api.py
├── scripts/
│   ├── generate_data.py         # CLI wrapper
│   ├── train_model.py           # CLI wrapper
│   └── serve_api.py             # runs FastAPI app
├── data/                        # dataset outputs
│   └── iris.csv
├── models/                      # trained models
│   └── iris_model.pkl
├── airflow_home/                # Airflow metadata (created after setup)
├── requirements.txt             # Python dependencies
└── setup_airflow.sh             # one-time setup script
```

---

## 🛠 Environment Setup

We use **one virtual environment** for all labs.

1. Create and activate:

```
python3 -m venv ~/venvs/airflow-class
source ~/venvs/airflow-class/bin/activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

⚠️ The `requirements.txt` pins **Airflow 2.10.2**. If you are not on Python 3.10, update the constraints line to match (`constraints-3.9.txt` or `constraints-3.11.txt`).  

---

## ⚙️ Airflow Setup (one time)

Run the setup script:

```
./setup_airflow.sh
```

This will:  
- Set `AIRFLOW_HOME` inside this project.  
- Initialize the Airflow database.  
- Create an admin user (`admin / admin`).  
- Symlink your `dags/` folder into Airflow’s DAGs directory.  

Afterwards, open a new terminal (or `source ~/.bashrc` / `~/.zshrc`) so `$AIRFLOW_HOME` is available automatically.  

---

## 🚀 Running Airflow

Use two terminals:

**Terminal 1 – Scheduler**
```
source ~/venvs/airflow-class/bin/activate
airflow scheduler
```

**Terminal 2 – Webserver**
```
source ~/venvs/airflow-class/bin/activate
airflow webserver --port 8080 --host 0.0.0.0
```

Then visit 👉 http://<ipaddress>:8080  
Login: `admin / admin`

Replace `<ipaddress>` with your EC2 instance’s **public IPv4 address**.  

---

## 📊 DAGs to Explore

You will see three DAGs:

**`ml pipeline dag`**  -->
- this includes 3 tasks namely loading  `data/breast_cancer.csv`. dataset, training the model on the dataset, evaluating the model and if its good its promoted and uploaded to an s3 bucket-->


<!--1. **`generate_data_only`**  -->
<!--   - Saves `data/iris.csv`.-->

<!--2. **`train_model_only`**  -->
<!--   - Trains a Logistic Regression model from CSV.  -->
<!--   - Produces `models/iris_model.pkl`.-->

<!--3. **`ml_pipeline`**  -->
<!--   - End-to-end pipeline:  -->
<!--     `generate_data` → `train_model`.-->

---

## 🧪 Testing Without Airflow

You can also run scripts directly:

```
python scripts/serve_api.py
python scripts/train_model.py
```

This will produce `data/breast_cancer.csv` and `models/breast_cancer_model.pkl`.  

---

## 🌐 Serving the Model with FastAPI

After training the model, you can serve it with FastAPI.

1. Run the API:

```
python scripts/serve_api.py
```

2. Open docs: http://<ipaddress>:8000/docs  

3. Try a prediction in Swagger UI with the  required features:

```
{
   mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float
}
```

Response:

```
{"prediction": "negative", "class_index": 0}
```

---

## 🌸 Example Inputs


---

## ✅ Summary

By the end of this lab you will have:  
- Built a training pipeline with Airflow.  
- Produced a dataset and a trained model artifact.  
- Served the trained model with FastAPI.  
- Sent live inference requests to your model.  

Next steps: containerize this API and deploy it to the cloud 🚀
