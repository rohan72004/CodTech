import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
# Load a sample dataset
df = px.data.iris()
# Start Dash app
app = dash.Dash(__name__)
# Layout
app.layout = html.Div([
    html.H1("Interactive Iris Dataset Dashboard", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Select Species:"),
        dcc.Dropdown(
            id='species-dropdown',
            options=[{'label': sp, 'value': sp} for sp in df['species'].unique()],
            value='setosa',
            clearable=False
        )
    ], style={'width': '30%', 'display': 'inline-block'}),
    dcc.Graph(id='scatter-plot'),
    html.Div([
        html.H4("Dataset Preview"),
        dcc.Markdown("First 5 Rows of Filtered Data"),
        html.Div(id='table-container')
    ])
])
# Callbacks
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('table-container', 'children')],
    [Input('species-dropdown', 'value')]
)
def update_graph(selected_species):
    filtered_df = df[df['species'] == selected_species]
    fig = px.scatter(
        filtered_df,
        x='sepal_width',
        y='sepal_length',
        color='species',
        title=f'Sepal Width vs Length: {selected_species}'
    )
    table = filtered_df.head().to_markdown(index=False)
    return fig, f"```\n{table}\n```"
# Run app
if __name__ == '__main__':
    app.run(debug=True)
