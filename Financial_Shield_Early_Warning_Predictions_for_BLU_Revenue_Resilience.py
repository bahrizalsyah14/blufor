
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Persiapan Data
# Baca data dari file CSV
data = pd.read_csv('https://github.com/bahrizalsyah14/blufor/blob/master/data_blu.csv', sep=';')

# Fitur yang akan digunakan untuk prediksi (variabel independen)
features = ['BI_Rate', 'Inflasi_maret', 'Tingkat_Kemiskinan', 'Tingkat_Pengangguran', 
            'Suku_Bunga', 'PDRB', 'Kebijakan_fiskal', 'Pendapatan_tengah', 'Pertumbuhan_ekonomi']

# X adalah fitur (variabel independen)
X = data[features]

# y adalah target (variabel dependen) yaitu Pendapatan_akhir
y = data['Pendapatan_akhir']

# Normalisasi fitur menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Pembuatan Model Machine Learning
# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning dengan GridSearchCV untuk RandomForestRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Menggunakan model terbaik dari hasil tuning
best_model = grid_search.best_estimator_

# Latih kembali model terbaik dengan data training
best_model.fit(X_train, y_train)

# Evaluasi model terbaik menggunakan cross-validation
scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
mean_score = np.mean(scores)
print(f'Mean Cross-Validation Score after Tuning: {-mean_score:.2f}')

# 3. Membangun Aplikasi Dash
# Inisialisasi aplikasi Dash
app = Dash(__name__)

# Layout aplikasi
app.layout = html.Div(children=[
    html.H1(children='Prediksi Optimalisasi Pendapatan BLU dengan Analisis Data Interaktif', style={'color': '#2E86C1'}),

    # Membuat input angka untuk setiap fitur
    *[html.Div([
        html.Label(feature, style={'color': '#2874A6'}),
        dcc.Input(
            id=f'{feature}-input',
            type='number',
            value=0,
            step=0.1,
            style={'margin-bottom': '10px', 'padding': '5px', 'border-radius': '5px', 'border': '1px solid #1F618D'}
        )
    ]) for feature in features],

    html.Hr(style={'border': '1px solid #AED6F1'}),
    html.Div(id='prediksi-output', style={'color': '#1B4F72', 'font-weight': 'bold', 'font-size': '20px'}),
    html.Button('Restart', id='restart-button', n_clicks=0, style={'margin-top': '20px', 'padding': '10px', 'background-color': '#2E86C1', 'color': 'white', 'border': 'none', 'border-radius': '5px'})
])

# Callback untuk memperbarui prediksi berdasarkan input user
@app.callback(
    Output('prediksi-output', 'children'),
    [Input(f'{feature}-input', 'value') for feature in features]
)
def update_output(*input_values):
    # Update data dengan nilai fitur baru dari input
    input_data = pd.DataFrame([input_values], columns=features)
    input_data_scaled = scaler.transform(input_data)

    # Prediksi menggunakan model
    if all(value == 0 for value in input_values):
        formatted_prediction = "0"
    else:
        prediction = best_model.predict(input_data_scaled)[0]
        # Menjamin prediksi tidak kurang dari Pendapatan_tengah
        pendapatan_tengah = input_data['Pendapatan_tengah'].values[0]
        if prediction < pendapatan_tengah:
            prediction = pendapatan_tengah
        # Format prediksi dengan pemisah ribuan dan desimal
        formatted_prediction = f"{prediction:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

    # Tampilkan prediksi
    prediksi_output = html.P(f"Prediksi Pendapatan Akhir: {formatted_prediction}", style={'color': '#0B5345'})

    return prediksi_output

# Callback untuk reset input ke default
@app.callback(
    [Output(f'{feature}-input', 'value') for feature in features],
    [Input('restart-button', 'n_clicks')]
)
def reset_inputs(n_clicks):
    if n_clicks > 0:
        return [0 for _ in features]
    return [0 for _ in features]

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run_server(debug=True)
