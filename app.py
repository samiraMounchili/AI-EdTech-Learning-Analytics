from flask import Flask, render_template, request, redirect, url_for, session
from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-demo-key")

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "data" / "micro_mba_dataset_final.csv")

module_columns = {
    "Strategy": "strategy_score",
    "Leadership": "leadership_score",
    "Sustainability": "sustainability_score",
    "Finance": "finance_score",
    "Operations": "operations_score",
    "Marketing": "marketing_score",
    "Digital Transformation": "digital_transformation_score"
}

VIDEO_LINK = "https://www.youtube.com/playlist?list=PL51qSDx1fblh7g2iVh-wNG1vQvQEle3YC"
COURSE_LINK = "https://www.chester.ac.uk/study/micro-mba/"

X = df.drop(columns=["completion_status", "behaviour_score", "learning_preference"], errors="ignore")
X = X.select_dtypes(include=[np.number])
y = df["completion_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


def get_risk(probability, progress=None, weakest_score=None):

    if progress >= 75:
        return "Low Risk"

    elif progress >= 60:
        return "Medium Risk"

    else:
        return "High Risk"


def get_learner():
    learner = df.iloc[0].to_dict()
    updates = session.get("learner_updates", {})
    learner.update(updates)
    return learner


def predict_probability(learner):
    row = {col: learner.get(col, 0) for col in X.columns}
    row_df = pd.DataFrame([row])
    return model.predict_proba(row_df)[0][1]


def build_modules(learner):
    modules = []

    for name, col in module_columns.items():
        score = int(learner[col])

        if score >= 80:
            status = "Completed"
            priority = "Strong Area"
        elif score >= 60:
            status = "In Progress"
            priority = "Medium Priority"
        else:
            status = "Needs Focus"
            priority = "High Priority"

        modules.append({
            "name": name,
            "score": score,
            "status": status,
            "priority": priority,
            "video": VIDEO_LINK,
            "course": COURSE_LINK
        })

    return modules


def build_stats(learner, modules):
    completed_courses = sum(1 for m in modules if m["score"] >= 80)
    quiz_attempts = int(learner.get("quiz_attempts", 0))
    time_spent = int(learner.get("time_spent_minutes", 0))

    return {
        "questions_solved": quiz_attempts * 7,
        "courses_completed": completed_courses,
        "certificates": 1 if completed_courses >= 4 else 0,
        "hours_studied": round(time_spent / 60, 1)
    }


def build_action_plan(modules, latest_update=None):
    weakest = sorted(modules, key=lambda x: x["score"])[0]

    if latest_update:
        module = latest_update["module"]
        old_score = latest_update["old_score"]
        new_score = latest_update["new_score"]
        change = new_score - old_score

        if new_score >= 80:
            return {
                "title": "Target Achieved 🎉",
                "message": f"{module} improved from {old_score}% to {new_score}%.",
                "action": f"You can now focus on {weakest['name']}, which is currently your weakest module.",
                "target": "Maintain 80%+"
            }

        elif change > 0:
            return {
                "title": "Progress Detected 📈",
                "message": f"{module} improved from {old_score}% to {new_score}%.",
                "action": f"Continue revising {module}, review the examples, and attempt another quiz.",
                "target": "80%"
            }

        else:
            return {
                "title": "Additional Support Needed",
                "message": f"{module} changed from {old_score}% to {new_score}%.",
                "action": f"Rewatch the {module} session, review the case study, and contact the tutor if you still feel unsure.",
                "target": "80%"
            }

    return {
        "title": f"Focus on {weakest['name']}",
        "message": f"Your lowest score is {weakest['score']}% in {weakest['name']}.",
        "action": f"Watch the {weakest['name']} content, review examples, and retake the quiz.",
        "target": "80%"
    }


def chatbot_answer(question, learner, modules, probability):
    q = question.lower()
    weakest = sorted(modules, key=lambda x: x["score"])[0]
    strongest = sorted(modules, key=lambda x: x["score"], reverse=True)[0]

    progress = round(sum(m["score"] for m in modules) / len(modules), 1)
    risk = get_risk(probability, progress, weakest["score"])

    tutor_contacts = {
        "Strategy": "For Strategy support, contact your module tutor or programme team through Moodle.",
        "Leadership": "For Leadership support, contact your module tutor or programme team through Moodle.",
        "Sustainability": "For Sustainability support, contact your module tutor or programme team through Moodle.",
        "Finance": "For Finance support, contact your Finance lecturer or programme team through Moodle.",
        "Operations": "For Operations support, contact your module tutor or programme team through Moodle.",
        "Marketing": "For Marketing support, contact your module tutor or programme team through Moodle.",
        "Digital Transformation": "For Digital Transformation support, contact your module tutor or programme team through Moodle."
    }

    rest_words = ["relax", "rest", "break", "tired", "sleep", "burnout", "overwhelmed"]
    negative_words = ["stress", "stressed", "worried", "confused", "struggling", "lost", "anxious"]

    if any(word in q for word in rest_words):
        if int(learner.get("time_spent_minutes", 0)) > 240 or int(learner.get("engagement_level", 0)) >= 7:
            return (
                "Yes, taking a short break is reasonable 😊. "
                "You have already spent a good amount of time learning. "
                f"After resting, return to your focus module: {weakest['name']}."
            )
        return (
            "A short break is okay 😊, but try one small learning task first. "
            f"Your focus module is {weakest['name']} with {weakest['score']}%. "
            "Watch 10 minutes of the lesson, then rest."
        )

    if any(word in q for word in negative_words):
        return (
            "I understand 😊. You do not need to fix everything at once. "
            f"Your main focus area is {weakest['name']} with {weakest['score']}%. "
            "Start with one step: rewatch the session, review examples, then retake the quiz. "
            + tutor_contacts.get(weakest["name"], "")
        )

    for module in modules:
        if module["name"].lower() in q:
            score = module["score"]

            if score >= 80:
                return (
                    f"You are doing well in {module['name']} with {score}% 🎉. "
                    f"Your strongest area is {strongest['name']}. "
                    f"You can now focus more on {weakest['name']}."
                )
            elif score >= 60:
                return (
                    f"Your {module['name']} score is {score}%, so you are close to the 80% target. "
                    "Review the examples or case study, then retake the quiz. "
                    + tutor_contacts.get(module["name"], "")
                )
            else:
                return (
                    f"{module['name']} needs more attention because your score is {score}%. "
                    "Rewatch the full session, revise for 45–60 minutes, and retake the quiz. "
                    + tutor_contacts.get(module["name"], "")
                )

    if "risk" in q:
        return (
            f"Your current risk level is {risk}. "
            "This is based on your overall progress, weakest module score, engagement, quiz attempts, time spent, and completion probability. "
            f"Your current focus area is {weakest['name']} at {weakest['score']}%."
        )

    if "recommend" in q or "study" in q or "focus" in q or "next" in q:
        return (
            f"I recommend focusing on {weakest['name']} first because it is your lowest score at {weakest['score']}%. "
            "Your next best action is to watch the module content, review examples, and retake the quiz with a target of 80%."
        )

    if "contact" in q or "lecturer" in q or "tutor" in q or "help" in q:
        return (
            f"For academic help, start with support for {weakest['name']}. "
            + tutor_contacts.get(weakest["name"], "Contact your module tutor or programme team through Moodle.")
        )

    return (
        "I’m here to help 😊. You can ask me things like: "
        "'What should I study next?', 'Can I relax?', 'Help me with Finance', "
        "'Why am I at risk?', or 'Who should I contact?'"
    )


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["user"] = request.form["email"]
        session["learner_updates"] = {}
        session["latest_update"] = None
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    instant_feedback = None
    chat_response = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "update":
            learner_before = get_learner()
            before_prob = predict_probability(learner_before)

            updates = session.get("learner_updates", {})

            updates["sessions_viewed"] = int(request.form["sessions_viewed"])
            updates["time_spent_minutes"] = int(request.form["time_spent_minutes"])
            updates["moodlet_logins"] = int(request.form["moodlet_logins"])
            updates["quiz_attempts"] = int(request.form["quiz_attempts"])
            updates["engagement_level"] = int(request.form["engagement_level"])

            selected_module = request.form["module"]
            new_score = int(request.form["new_score"])

            score_col = module_columns[selected_module]
            old_score = int(learner_before[score_col])

            updates[score_col] = new_score

            session["learner_updates"] = updates
            session["latest_update"] = {
                "module": selected_module,
                "old_score": old_score,
                "new_score": new_score
            }
            session.modified = True

            learner_after = get_learner()
            after_prob = predict_probability(learner_after)
            change = round((after_prob - before_prob) * 100, 2)

            if new_score >= 80:
                instant_feedback = (
                    f"Excellent progress 😊 {selected_module} improved from {old_score}% to {new_score}%. "
                    f"Completion probability changed by {change}%."
                )
            elif new_score >= 60:
                instant_feedback = (
                    f"Good improvement. {selected_module} changed from {old_score}% to {new_score}%. "
                    f"Completion probability changed by {change}%."
                )
            else:
                instant_feedback = (
                    f"{selected_module} is below target at {new_score}%. "
                    f"Review the content and retake the quiz. Completion probability changed by {change}%."
                )

        elif action == "chat":
            learner = get_learner()
            modules = build_modules(learner)
            probability = predict_probability(learner)
            chat_response = chatbot_answer(request.form["question"], learner, modules, probability)

    learner = get_learner()
    probability = predict_probability(learner)

    modules = build_modules(learner)
    progress = round(sum(m["score"] for m in modules) / len(modules), 1)

    strongest_module = max(modules, key=lambda x: x["score"])
    weakest_module = min(modules, key=lambda x: x["score"])

    risk = get_risk(probability, progress, weakest_module["score"])

    stats = build_stats(learner, modules)

    weakest_modules = sorted(modules, key=lambda x: x["score"])[:3]
    strongest_modules = sorted(modules, key=lambda x: x["score"], reverse=True)[:2]

    action_plan = build_action_plan(modules, session.get("latest_update"))

    return render_template(
        "dashboard.html",
        user=session["user"],
        learner=learner,
        probability=round(probability * 100, 2),
        risk=risk,
        modules=modules,
        stats=stats,
        progress=progress,
        strongest_module=strongest_module,
        weakest_module=weakest_module,
        weakest_modules=weakest_modules,
        strongest_modules=strongest_modules,
        action_plan=action_plan,
        instant_feedback=instant_feedback,
        chat_response=chat_response
    )


@app.route("/courses")
def courses():
    if "user" not in session:
        return redirect(url_for("login"))

    learner = get_learner()
    modules = build_modules(learner)

    return render_template("courses.html", modules=modules)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)