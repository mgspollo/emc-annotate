import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from src.model.identify_power_supply import predict_full
from src.data.import_data import normalise_spectra

max_test_id = 10080
# Sample data for the example
test_ids = [f"{i}" for i in range(10001, max_test_id)]
df = normalise_spectra(max_test_id)

app = dash.Dash(__name__)

# Assuming `df_metadata` is your metadata dataframe
metadata = predict_full()



app.layout = html.Div(style={'backgroundColor': '#f8f8f8', 'fontFamily': 'Arial, sans-serif'}, children=[
    html.Div(style={'padding': '20px', 'backgroundColor': '#3498db', 'color': 'white', 'textAlign': 'center'},
             children=[
                 html.Img(src="assets/kenlock_logo.PNG",
                          style={'width': '250px', 'height': 'auto', 'marginBottom': '10px'}),
                 html.H1("EMC Spectrum Component Detection Tool", style={'marginBottom': '5px'}),
             ]),

    html.Div(style={'display': 'flex', 'justifyContent': 'center', 'padding': '20px'}, children=[
        html.Div(style={
            'width': '30%', 'padding': '20px', 'backgroundColor': 'white',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'borderRadius': '10px',
            'marginRight': '20px'
        }, children=[
            html.Label("Select Test Spectrum(s)", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id="test-dropdown",
                options=[{"label": col, "value": col} for col in test_ids],
                value=[test_ids[0]],
                multi=True,
                style={'marginTop': '10px', 'marginBottom': '20px'}
            ),
            html.Div(id="metadata-info", style={'marginTop': '20px'})
        ]),

        html.Div(style={
            'width': '60%', 'padding': '20px', 'backgroundColor': 'white',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'borderRadius': '10px'
        }, children=[
            dcc.Graph(id="raw-ambient-plot"),
            dcc.Graph(id="normalized-plot")
        ])
    ]),
])


@app.callback(
    [Output("raw-ambient-plot", "figure"),
     Output("normalized-plot", "figure"),
     Output("metadata-info", "children")],
    [Input("test-dropdown", "value")]
)
def update_plots(selected_tests):
    raw_ambient_fig = go.Figure()
    normalized_fig = go.Figure()
    metadata_info_list = []

    for test_id in selected_tests:
        ambient_col = f"{test_id}a"
        normalized_col = f"{test_id}n"

        # Add raw and ambient traces to raw_ambient_fig
        raw_trace = go.Scatter(
            x=df.index,
            y=df[test_id], mode='lines', name=f'Raw Signal {test_id}'
        )
        ambient_trace = go.Scatter(
            x=df.index,
            y=df[ambient_col], mode='lines', name=f'Ambient Signal {test_id}'
        )
        raw_ambient_fig.add_trace(raw_trace)
        raw_ambient_fig.add_trace(ambient_trace)

        # Add normalized trace to normalized_fig
        normalized_trace = go.Scatter(
            x=df.index,
            y=df[normalized_col], mode='lines', name=f'Normalized Signal {test_id}'
        )
        normalized_fig.add_trace(normalized_trace)

        # Collect metadata for sidebar
        metadata_info_list.append(html.Div(
            style={'padding': '10px', 'border': '1px solid #dedede', 'borderRadius': '5px', 'marginBottom': '10px'},
            children=[
                html.P(f"Test ID: {test_id}", style={'fontWeight': 'bold'}),
                html.P(f"Product Description: {metadata.loc[test_id, 'product_description']}"),
                html.P(f"Power Supply: {'Yes' if metadata.loc[test_id, 'is_power_supply'] else 'No'}"),
                html.P(f"HDMI Present: {metadata.loc[test_id, 'is_HDMI']}"),
                html.P(f"Display Port Present: {metadata.loc[test_id, 'is_DisplayPort']}"),
                html.P("Predicted Components:", style={'fontWeight': 'bold'}),
                html.P(f"Predicted HDMI Present: {metadata.loc[test_id, 'is_hdmi_prediction']}"),
                html.P(f"Predicted DisplayPort Present: {metadata.loc[test_id, 'is_display_port_prediction']}"),
                html.P(f"Predicted Power Supply Present: {metadata.loc[test_id, 'is_power_supply_prediction']}"),
                html.Hr()
            ]))

    # Update layout for raw and ambient plot
    raw_ambient_fig.update_layout(
        title="Raw and Ambient Signals",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        plot_bgcolor='white',
        margin=dict(l=40, r=10, t=40, b=40)
    )

    # Update layout for normalized plot
    normalized_fig.update_layout(
        title="Normalized Signal",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        plot_bgcolor='white',
        margin=dict(l=40, r=10, t=40, b=40)
    )

    return raw_ambient_fig, normalized_fig, metadata_info_list


if __name__ == '__main__':
    app.run_server(debug=True)
