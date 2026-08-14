# AI EdTech Learning Analytics

A Flask-based learning analytics prototype that combines **machine learning, learner progress analytics, personalised recommendations and a supportive chatbot** to demonstrate how short-course learners could receive more targeted support.

## Project overview

This project was developed as an academic prototype around a Micro MBA learning context. It explores how learner behaviour and module performance can be used to estimate completion likelihood, identify areas that need attention and generate personalised learning actions.

The repository brings together two parts of the project:

- **Data and machine learning:** synthetic learner-behaviour data, Logistic Regression and model evaluation.
- **Interactive web prototype:** a Flask dashboard with learner analytics, module progress, adaptive recommendations and a rule-based support chatbot.

> **Data and evaluation note:** The dataset is synthetic. Any model accuracy reported in the notebook reflects performance on synthetically generated data and is not evidence of validated performance on real students.

> **Project-status note:** This is an academic demonstration, not an official University system or a production authentication platform.

## Key features

- Completion-probability estimation using Logistic Regression
- Learner progress and engagement dashboard
- Module-level performance tracking
- Identification of strongest and weakest learning areas
- Personalised learning action recommendations
- Dynamic feedback after quiz-score updates
- Adaptive content recommendations
- Rule-based, sentiment-aware learner support chatbot
- Demo login and learner profile interface
- Synthetic dataset containing 612 learner records

## Technologies

- Python
- Flask
- pandas
- NumPy
- scikit-learn
- HTML
- CSS
- Jupyter Notebook

## Repository structure

```text
AI-EdTech-Learning-Analytics/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── micro_mba_dataset_final.csv
├── notebooks/
│   └── learner_completion_analysis.ipynb
├── templates/
│   ├── login.html
│   ├── dashboard.html
│   └── courses.html
└── static/
    └── style.css
```

## Dataset

The synthetic dataset contains 612 records and includes variables such as:

- sessions viewed
- time spent learning
- Moodle logins
- days since last login
- engagement level
- quiz attempts
- rewatched sessions
- module scores across Strategy, Leadership, Sustainability, Finance, Operations, Marketing and Digital Transformation
- completion status
- learning preference

No real student personal data is included.

## Machine-learning approach

The notebook creates a synthetic learner dataset and trains a **Logistic Regression** model to estimate completion probability. The analysis uses an 80/20 train/test split with a fixed random state for reproducibility.

The recorded notebook run achieved approximately **96.7% test accuracy**. Because the target variable is derived from rules applied to synthetic behaviour data, this result should be interpreted as a prototype demonstration rather than a real-world validation result.

## Personalisation logic

The prototype combines model output and learner progress to provide:

- a completion probability
- a learner risk band
- identification of the weakest module
- recommended next actions
- target quiz scores
- adaptive learning-content suggestions

The chatbot also uses simple keyword and sentiment rules to respond to questions about study priorities, academic support, risk and learner wellbeing.

## Running the application locally

1. Clone or download the repository.
2. Create a virtual environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Optionally set a Flask secret key in your environment:

```bash
set FLASK_SECRET_KEY=your-local-development-key
```

5. Run:

```bash
python app.py
```

6. Open the local address shown by Flask, normally `http://127.0.0.1:5000`.

The login screen is for demonstration only. It does not implement production authentication.

## Limitations

- The dataset is synthetic rather than collected from real learners.
- Completion status is generated from predefined behavioural rules.
- The model therefore requires validation on ethically approved real-world data before any operational use.
- The chatbot is rule-based rather than a generative AI system.
- The login page is a prototype interface and does not authenticate against a real identity service.
- The Flask application is intended for local demonstration and portfolio use, not production deployment.

## Future improvements

- Validate the modelling approach using ethically approved anonymised learner data.
- Compare additional models using cross-validation and class-imbalance-aware metrics.
- Add explainability so learners and staff can understand why a prediction changed.
- Add secure authentication and persistent database storage.
- Expand module-specific recommendation logic.
- Improve accessibility and responsive design.
- Add model monitoring and fairness checks.

## Portfolio relevance

This project demonstrates practical experience in:

**Data analysis • Machine learning • Python • Flask • Data visualisation • Business problem solving • User-centred analytics • Responsible use of synthetic data**
