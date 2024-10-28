import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Sample data for the sake of the example (replace with actual test spectra and predictions)
# Assuming `X_test` contains test spectra features and `scores` holds detection likelihoods

# Sample DataFrame setup
X_test = pd.DataFrame(np.random.rand(10, 100),
                      index=[f"Test_{i}" for i in range(10)])  # Replace with actual test spectra
scores = {
    'HDMI_present': np.random.rand(10),
    'DisplayPort_present': np.random.rand(10),
    'PowerSupply_present': np.random.rand(10)
}

# Initialize the app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("EMC Spectrum Component Detection"),

    # Dropdown for test selection
    html.Label("Select Test Spectrum"),
    dcc.Dropdown(
        id="test-dropdown",
        options=[{"label": idx, "value": idx} for idx in X_test.index],
        value=X_test.index[0]
    ),

    # Detection Scores Display
    html.Div(id="detection-scores", style={'margin-top': '20px'}),

    # Spectrum Plot
    dcc.Graph(id="spectrum-plot")
])


# Callback for updating scores and plot based on selection
@app.callback(
    [Output("detection-scores", "children"),
     Output("spectrum-plot", "figure")],
    [Input("test-dropdown", "value")]
)
def update_display(selected_test):
    # Fetch detection likelihood scores for the selected test
    hdmi_score = scores['HDMI_present'][X_test.index.get_loc(selected_test)]
    dp_score = scores['DisplayPort_present'][X_test.index.get_loc(selected_test)]
    ps_score = scores['PowerSupply_present'][X_test.index.get_loc(selected_test)]

    # Display scores
    score_display = [
        html.P(f"HDMI Likelihood: {hdmi_score:.2f}"),
        html.P(f"DisplayPort Likelihood: {dp_score:.2f}"),
        html.P(f"Power Supply Likelihood: {ps_score:.2f}")
    ]

    # Spectrum Plot
    spectrum_data = X_test.loc[selected_test]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(spectrum_data))), y=spectrum_data,
        mode='lines', name='Spectrum'
    ))

    # Highlight likelihoods for easy visualization
    fig.update_layout(
        title=f"Spectrum for {selected_test}",
        xaxis_title="Frequency",
        yaxis_title="Magnitude"
    )

    return score_display, fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
