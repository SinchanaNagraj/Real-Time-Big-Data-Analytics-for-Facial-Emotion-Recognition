# Real-Time Big Data Analytics for Facial Emotion Recognition

## Quick start (Windows / VS Code)

1. Install Python 3.10 or 3.11 and open this folder in VS Code.
2. Open Terminal (`Ctrl + Shift + ``) and run:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this in the same terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

3. The browser dashboard opens. First model use may download pretrained weights and take longer.
4. Try **Upload image**, **Webcam snapshot**, then a short **Upload video**.
5. Open **Analytics dashboard** to view records and download CSV.

## What is implemented
- Pretrained DeepFace emotion analysis (seven expression labels)
- Image upload and browser webcam snapshot input
- Video frame sampling with OpenCV
- Timestamped prediction event storage in SQLite
- Pandas/Plotly distribution and time-trend analytics
- CSV downloads

## Dataset note
This quick-deadline version uses a pretrained model for inference, so a separate training dataset is not required to run the app. If the submission rubric specifically requires a dataset, use the FER-2013 dataset as an **offline benchmark / dataset-analysis extension**; do not claim that this project trained the model on FER-2013 unless you actually train and evaluate it. FER-2013 is commonly distributed through Kaggle and requires account/license acceptance. Download it from Kaggle's FER-2013 dataset page, then keep the downloaded data in a `dataset/` folder (do not upload private face images).

## Honest project scope
This is a small local real-time analytics prototype, not a distributed big-data cluster. SQLite and sampled video demonstrate event storage and stream-like processing. A true production big-data implementation could add Kafka, Spark Streaming, and distributed storage.

## Responsible use
Facial-expression labels can be inaccurate and are not reliable proof of a person's internal emotional state. Obtain consent before recording or processing people.
