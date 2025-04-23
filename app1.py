import dash
from dash import html, dcc, Input, Output, State, dash_table
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load model and data
best_model = joblib.load("../artifacts/model.pkl")
data = pd.read_csv("../data/movie.csv")
data_sample = data.sample(20, random_state=42).copy()
data_sample['predicted'] = best_model.predict(data_sample['text'])

# Calculate metrics
X_test = data['text']
y_true = data['label']
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Movie Review Sentiment Classifier (Naive Bayes)"),

    html.Div([
        html.H3("Model Performance Metrics"),
        html.Ul([
            html.Li(f"Accuracy: {accuracy:.2f}"),
            html.Li(f"Precision: {precision:.2f}"),
            html.Li(f"Recall: {recall:.2f}"),
            html.Li(f"F1 Score: {f1:.2f}"),
        ])
    ], style={'marginBottom': 30}),

    html.Div([
        dcc.Textarea(
            id='input-text',
            placeholder='Enter a movie review...',
            style={'width': '100%', 'height': 150}
        ),
        html.Br(),
        html.Button('Predict Sentiment', id='submit-button', n_clicks=0),
        html.Div(id='prediction-output', style={'marginTop': 20, 'fontSize': 20})
    ], style={'marginBottom': 50}),

    html.H3("Sample Reviews and Predicted Sentiment"),
    dash_table.DataTable(
        id='sample-table',
        columns=[
            {'name': 'Review', 'id': 'text'},
            {'name': 'Actual Label', 'id': 'label'},
            {'name': 'Predicted Label', 'id': 'predicted'}
        ],
        data=data_sample.to_dict('records'),
        style_table={'height': '300px', 'overflowY': 'auto'},
        style_cell={
            'textAlign': 'left',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        page_size=20
    )
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-text', 'value')
)
def predict_sentiment(n_clicks, input_text):
    if n_clicks > 0 and input_text:
        prediction = best_model.predict([input_text])[0]
        return f"Predicted Sentiment: {prediction}"
    return ""

if __name__ == '__main__':
    app.run(debug=True)
