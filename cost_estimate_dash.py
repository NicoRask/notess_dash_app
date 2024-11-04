from cost_utils import LLMSystem
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px  # Optional, for graphing if needed

# Initialize Dash app
app = dash.Dash(__name__)



# load the list with customer_profiles
inpath = "C:/Users/NicolaiRaskMathiesen/OneDrive - Rooftop Analytics ApS/Desktop/"
fname = "customer_profiles.xlsx"
customer_profiles = pd.read_excel(inpath + fname, index_col=0)
customer_profiles["files"] = customer_profiles[["pages", "audio_min", "images"]].sum(axis=1)

components = {
    "LLM": "GPT-4o_mini",
    "embedding": "GPT_text-embedding-3-small",
    "audio": "Azure AI Speech",
    "image_captioning": "Azure AI vision API",
    "face_recognition": "Azure AI vision Face API",
    "RAG": "pinecone",
    "blob": "azure hot",
}

llm_sys = LLMSystem()
data_intro_costs, monthly_costs = llm_sys.get_overall_costs(components, customer_profiles.loc["customer1"])

# Layout of Dash app
app.layout = html.Div([
    html.H1("System Implementation Cost Calculator"),

    # Dropdown to select customer profile
    html.Label("Select Customer Profile"),
    dcc.Dropdown(
        id='profile-dropdown',
        options=[{'label': profile, 'value': i} for i, profile in enumerate(customer_profiles.index)],
        value=0  # Default to first profile
    ),

    # Sliders for adjustable constants
    #html.Label("Storage Cost per GB"),
    #dcc.Slider(id='storage-cost-slider', min=0, max=0.1, step=0.01, value=DEFAULT_STORAGE_COST),

    #html.Label("API Call Cost"),
    #dcc.Slider(id='api-cost-slider', min=0, max=0.1, step=0.01, value=DEFAULT_API_CALL_COST),

    #html.Label("Data Conversion Rate"),
    #dcc.Slider(id='conversion-rate-slider', min=0, max=0.1, step=0.01, value=DEFAULT_CONVERSION_RATE),

    # Div to display results
    html.Div(id='output-div', style={'margin-top': '20px'})
])


# Callback to update output based on inputs
@app.callback(
    Output('output-div', 'children'),
    Input('profile-dropdown', 'value'),
)
def update_output(profile_index):
    # Fetch selected profile data
    profile = customer_profiles.iloc[profile_index]
    # Calculate costs using selected profile and slider values
    costs = llm_sys.get_overall_costs(components, profile)
    print(costs[0], costs[1])
    # Display costs in a readable format
    return html.Div([
        html.H2(f"Cost Breakdown for {profile.name}"),
        html.P(f"Data intro Cost: ${costs[1]:.2f}"),
        html.P(f"Monthly Cost: ${costs[0]:.2f}"),
        #html.P(f"Conversion Cost: ${costs['Conversion Cost']:.2f}"),
        #html.H3(f"Total Cost: ${costs['Total Cost']:.2f}")
    ])


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
