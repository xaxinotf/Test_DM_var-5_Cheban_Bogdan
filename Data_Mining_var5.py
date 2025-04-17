import pandas as pd
import numpy as np
import random
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

"""
Data_Mining_var5.py

Швидко: робимо K-Means++ і малюємо в Dash.
"""

def kmeans_pp_init(X, k):
    # k-means++: перший центр — випадковий
    centers = [X[random.randrange(len(X))]]
    for _ in range(1, k):
        # квадрат відстаней до найближчого центру
        dist_sq = np.array([min(np.sum((x - c)**2) for c in centers) for x in X])
        probs = dist_sq / dist_sq.sum()
        r = random.random()
        cum = np.cumsum(probs)
        centers.append(X[np.searchsorted(cum, r)])
    return np.array(centers)


def assign_clusters(X, centers):
    # кожну точку у найближчий кластер
    return np.array([np.argmin([np.sum((x - c)**2) for c in centers]) for x in X])


def update_centers(X, labels, k):
    # нові центри — середні по кластеру
    _, m = X.shape
    C = np.zeros((k, m))
    for i in range(k):
        pts = X[labels == i]
        if len(pts): C[i] = pts.mean(axis=0)
    return C


def compute_sse(X, centers, labels):
    # сума квадратів відхилень
    return sum(np.sum((x - centers[l])**2) for x, l in zip(X, labels))


def kmeans(X, k, max_iter=100):
    # основний цикл K-Means
    centers = kmeans_pp_init(X, k)
    for _ in range(max_iter):
        labels = assign_clusters(X, centers)
        new_centers = update_centers(X, labels, k)
        if np.allclose(new_centers, centers): break
        centers = new_centers
    return centers, labels

# --- Зчитування даних ---
df = pd.read_csv('synthetic_customers.csv', header=None,
                 names=['age','income','loyalty'], skiprows=1)
df = df.apply(pd.to_numeric)
X = df[['age','income','loyalty']].values

# --- Elbow-метод ---
sse = []
ks = range(1, 11)
for k in ks:
    c, lbl = kmeans(X, k)
    sse.append(compute_sse(X, c, lbl))
# знайти "лікть"
d2 = np.diff(sse, 2)
opt_k = int(np.argmax(d2) + 2)
print(f"Оптимальне k (метод ліктя): {opt_k}")

# --- Dash UI ---
app = dash.Dash(__name__)
app.index_string = '''<!DOCTYPE html><html><head>{%metas%}
<title>Кластеризація K-Means++</title><style>
 body {font-family:Arial,sans-serif;background:#f0f2f5;margin:0;padding:0}
 h1{text-align:center;color:#333;margin:20px 0}
 .cont{width:80%;margin:auto}
 .graf{background:#fff;padding:20px;border-radius:10px;box-shadow:0 2px 4px rgba(0,0,0,0.1);margin-bottom:30px}
 .slide{margin:20px 0}
</style>{%favicon%}{%css%}</head><body>{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=list(ks), y=sse, mode='lines+markers', name='SSE'))
fig1.update_layout(title='Графік ліктя: SSE від k',
                   xaxis_title='k', yaxis_title='SSE', plot_bgcolor='#f9f9f9')

app.layout = html.Div(className='cont', children=[
    html.H1('K-Means++ Кластеризація'),
    html.Div(className='graf', children=[dcc.Graph(id='elbow', figure=fig1)]),
    html.Div(className='slide', children=[
        html.Label(f'Оберіть k (рекомендація ~ {opt_k}):'),
        dcc.Slider(id='slider', min=1, max=10, step=1, value=opt_k,
                   marks={i:str(i) for i in ks})
    ]),
    html.Div(className='graf', children=[dcc.Graph(id='scatter')])
])

@app.callback(Output('scatter', 'figure'), Input('slider', 'value'))
def draw(k):
    C, L = kmeans(X, k)
    fig = go.Figure()
    for i in range(k):
        pts = X[L==i]
        fig.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1], mode='markers',
            name=f'Кластер {i+1}', marker_size=pts[:,2]/2))
    fig.add_trace(go.Scatter(x=C[:,0], y=C[:,1], mode='markers',
        name='Центроїди', marker=dict(symbol='x', size=12)))
    fig.update_layout(title=f'Візуалізація (k={k})', xaxis_title='Вік',
                      yaxis_title='Дохід', plot_bgcolor='#f9f9f9')
    return fig

if __name__ == '__main__':
    app.run(debug=True)
