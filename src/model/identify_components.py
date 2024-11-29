from src.data.import_data import normalise_spectra, process_metadata, filter_only_tests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

max_test_id = 10080

def add_features_metadata():
    df_metadata = process_metadata(max_test_id)

    # Assuming df_metadata is your metadata dataframe
    df_metadata['is_power_supply'] = df_metadata['power_supply_loading'].apply(lambda x: True if x in [0.5, 1] else False)

    # Define is_HDMI and is_DisplayPort based on display_output_protocol and display_receive_protocol
    df_metadata['is_HDMI'] = (
        ((df_metadata['display_output_protocol'] == 'HDMI') &
         (df_metadata['display_receive_protocol'] == 'HDMI')) |
        ((df_metadata['display_output_protocol'] == 'HDMI') &
         (df_metadata['display_receive_protocol'] == 'DisplayPort')) |
        ((df_metadata['display_output_protocol'] == 'DisplayPort') &
         (df_metadata['display_receive_protocol'] == 'HDMI'))
    )


    df_metadata['is_DisplayPort'] = (
        ((df_metadata['display_output_protocol'] == 'DisplayPort') &
         (df_metadata['display_receive_protocol'] == 'DisplayPort')) |
        ((df_metadata['display_output_protocol'] == 'HDMI') &
         (df_metadata['display_receive_protocol'] == 'DisplayPort')) |
        ((df_metadata['display_output_protocol'] == 'DisplayPort') &
         (df_metadata['display_receive_protocol'] == 'HDMI'))
    )
    return df_metadata


# Define frequency ranges for feature extraction
def extract_features(spectrum_df):
    features = {}
    # Example frequency ranges in Hz
    power_band = (100e6, 400e6)
    full_band = (30e6, 1e9)

    # Average magnitude and std deviation in power_band (for Power Supply)
    power_band_data = spectrum_df.loc[power_band[0]:power_band[1]]
    features['power_avg_magnitude'] = power_band_data.mean(axis=0)
    features['power_std_magnitude'] = power_band_data.std(axis=0)

    # Full spectrum stats (for HDMI and DisplayPort)
    full_band_data = spectrum_df.loc[full_band[0]:full_band[1]]
    features['full_avg_magnitude'] = full_band_data.mean(axis=0)
    features['full_std_magnitude'] = full_band_data.std(axis=0)

    # Count high peaks (arbitrary threshold, tune as needed)
    threshold = 0.8 * spectrum_df.max().max()
    features['peak_count'] = (spectrum_df > threshold).sum(axis=0)

    return pd.DataFrame(features)


def predict_full():
    # Prepare the dataset
    df_metadata = add_features_metadata()
    df_spectra = filter_only_tests(max_test_id)
    X = extract_features(df_spectra)  # Extract features from the spectra
    y_hdmi = df_metadata['is_HDMI']
    y_display_port = df_metadata['is_DisplayPort']
    y_power_supply = df_metadata['is_power_supply']

    # Train-Test Split for reproducibility
    X_train, X_test, y_hdmi_train, y_hdmi_test = train_test_split(X, y_hdmi, test_size=0.2)
    X_train, X_test, y_dp_train, y_dp_test = train_test_split(X, y_display_port, test_size=0.2)
    X_train, X_test, y_ps_train, y_ps_test = train_test_split(X, y_power_supply, test_size=0.2)

    # HDMI Model
    hdmi_model = LogisticRegression()
    hdmi_model.fit(X_train, y_hdmi_train)
    y_hdmi_pred = hdmi_model.predict(X_test)
    print("HDMI Model Accuracy:", accuracy_score(y_hdmi_test, y_hdmi_pred))

    # DisplayPort Model
    display_port_model = LogisticRegression()
    display_port_model.fit(X_train, y_dp_train)
    y_dp_pred = display_port_model.predict(X_test)
    print("DisplayPort Model Accuracy:", accuracy_score(y_dp_test, y_dp_pred))

    # Power Supply Model
    power_supply_model = LogisticRegression()
    power_supply_model.fit(X_train, y_ps_train)
    y_ps_pred = power_supply_model.predict(X_test)
    print("Power Supply Model Accuracy:", accuracy_score(y_ps_test, y_ps_pred))

    df_metadata["is_display_port_prediction"] = display_port_model.predict(X)
    df_metadata["is_hdmi_prediction"] = hdmi_model.predict(X)
    df_metadata["is_power_supply_prediction"] = power_supply_model.predict(X)
    return df_metadata

