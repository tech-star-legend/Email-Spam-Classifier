from flask import Flask, render_template, request
import pickle
import warnings

warnings.filterwarnings('ignore')

from utils.preprocess import transform_text

# -----------------------------------
# FLASK APP
# -----------------------------------

app = Flask(__name__)

# -----------------------------------
# LOAD MODEL + VECTORIZER
# -----------------------------------

vectorizer = pickle.load(
    open('model/vectorizer.pkl', 'rb')
)

svm_model = pickle.load(
    open('model/svm.pkl', 'rb')
)

# -----------------------------------
# SPAM KEYWORDS
# -----------------------------------

spam_keywords = [

    'free',
    'win',
    'winner',
    'offer',
    'money',
    'cash',
    'click',
    'buy',
    'urgent',
    'limited',
    'claim',
    'prize',
    'bonus'
]

# -----------------------------------
# PREDICTION HISTORY
# -----------------------------------

prediction_history = []

# -----------------------------------
# HOME ROUTE
# -----------------------------------

@app.route('/')
def home():

    return render_template(
        'index.html',
        history=prediction_history[::-1]
    )

# -----------------------------------
# DASHBOARD ROUTE
# -----------------------------------

@app.route('/dashboard')
def dashboard():

    return render_template(
        'dashboard.html'
    )

# -----------------------------------
# PREDICTION ROUTE
# -----------------------------------

@app.route('/predict', methods=['POST'])
def predict():

    email = request.form['email']

    # NLP preprocessing
    transformed_email = transform_text(email)

    # Vectorization
    vector_input = vectorizer.transform(
        [transformed_email]
    ).toarray()

    # Probability
    probability = svm_model.predict_proba(
        vector_input
    )[0]

    spam_probability = round(
        probability[1] * 100,
        2
    )

    ham_probability = round(
        probability[0] * 100,
        2
    )

    # Threshold logic
    if spam_probability >= 50:

        result = "SPAM"

    else:

        result = "HAM"

    # -----------------------------------
    # KEYWORD HIGHLIGHTING
    # -----------------------------------

    detected_keywords = []

    email_words = email.lower().split()

    for word in email_words:

        cleaned_word = word.strip(
            '.,!?()[]{}:;"\''
        )

        if cleaned_word in spam_keywords:

            if cleaned_word not in detected_keywords:

                detected_keywords.append(
                    cleaned_word
                )

    # -----------------------------------
    # SAVE HISTORY
    # -----------------------------------

    prediction_history.append({

        'email': email[:50] + "...",

        'result': result,

        'spam_probability': spam_probability
    })

    # Keep last 10 predictions
    if len(prediction_history) > 10:

        prediction_history.pop(0)

    # -----------------------------------
    # RETURN FRONTEND
    # -----------------------------------

    return render_template(

        'index.html',

        result=result,

        spam_probability=spam_probability,

        ham_probability=ham_probability,

        email=email,

        detected_keywords=detected_keywords,

        history=prediction_history[::-1]
    )

# -----------------------------------
# MAIN
# -----------------------------------

if __name__ == '__main__':

    app.run(debug=True)