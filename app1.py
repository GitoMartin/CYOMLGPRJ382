import dash
from dash import html, dcc, Input, Output, State, dash_table
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px

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

# Prepare pie chart with string labels
label_map = {0: 'Negative', 1: 'Positive'}
mapped_preds = pd.Series(y_pred).map(label_map)
sentiment_counts = mapped_preds.value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
pie_fig = px.pie(sentiment_counts, names='Sentiment', values='Count', title='Sentiment Prediction Distribution')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Movie Review Sentiment Classifier (Naive Bayes)"),

    html.Div([
        html.H3("Model Performance Metrics"),
        html.Ul([
            html.Li(f"Accuracy: {accuracy:.4f}"),
            html.Li(f"Precision: {precision:.4f}"),
            html.Li(f"Recall: {recall:.4f}"),
            html.Li(f"F1 Score: {f1:.4f}"),
        ])
    ], style={'marginBottom': 30}),
     html.H1("Model Evaluation Metrics", style={'textAlign': 'center'}),
    
    html.Div([
        html.H3("Accuracy", style={'fontWeight': 'bold'}),
        html.P("Accuracy measures the overall correctness of the model. It is the ratio of correctly predicted observations to the total observations."),
        html.P("Formula: Accuracy = (True Positives + True Negatives) / Total Predictions")
    ], style={'padding': '20px', 'border': '1px solid #ccc', 'margin': '10px'}),
    
    html.Div([
        html.H3("Precision", style={'fontWeight': 'bold'}),
        html.P("Precision measures how many of the predicted positives are actually positive."),
        html.P("Formula: Precision = True Positives / (True Positives + False Positives)")
    ], style={'padding': '20px', 'border': '1px solid #ccc', 'margin': '10px'}),
    
    html.Div([
        html.H3("Recall", style={'fontWeight': 'bold'}),
        html.P("Recall measures how many of the actual positives were correctly identified."),
        html.P("Formula: Recall = True Positives / (True Positives + False Negatives)")
    ], style={'padding': '20px', 'border': '1px solid #ccc', 'margin': '10px'}),
    
    html.Div([
        html.H3("F1 Score", style={'fontWeight': 'bold'}),
        html.P("F1 Score is the harmonic mean of precision and recall. It balances both metrics, especially in imbalanced datasets."),
        html.P("Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)")
    ], style={'padding': '20px', 'border': '1px solid #ccc', 'margin': '10px'}),

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
            {'name': 'Actual Sentiment', 'id': 'label'},
            {'name': 'Predicted Sentiment', 'id': 'predicted'}
        ],
        data=data_sample.to_dict('records'),
        style_table={'height': '300px', 'overflowY': 'auto'},
        style_cell={
            'textAlign': 'left',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        page_size=20
    ),

    html.H3("Sentiment Prediction Distribution"),
    dcc.Graph(figure=pie_fig)
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-text', 'value')
)
def predict_sentiment(n_clicks, input_text):
    if n_clicks > 0 and input_text:
        prediction = best_model.predict([input_text])[0]
        sentiment = label_map.get(prediction, str(prediction))
        return f"Predicted Sentiment: {sentiment}"
    return ""

if __name__ == '__main__':
    app.run(debug=True)
