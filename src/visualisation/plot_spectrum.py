from src.data.import_data import read_test_data
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns


# fig, ax = plt.subplots(figsize=(16, 16))
def plotting_signal_raw():
    df = read_test_data()
    df_all_ambient = pd.DataFrame()
    for i, row in df.iterrows():
        row = row.to_dict()
        # this is to plot every 5th element in the signal, so that plotly does not get overwhelmed with the number of
        # elements to plot
        df_signal = row["test_signal"].iloc[::5, :]
        df_ambient = row["test_ambient"].iloc[::5, :]
        if i == 0:
            df_all_ambient = df_ambient.rename(columns={"intensity": str(row["test_id"])})
            fig = px.line()
        else:
            df_all_ambient = pd.merge(df_all_ambient, df_ambient.rename(columns={"intensity": str(row["test_id"])}),
                                      on='frequency')
        fig.add_scatter(x=df_signal['frequency'], y=df_signal['intensity'], mode='lines', name=str(row["test_id"]))

    df_all_ambient = df_all_ambient.set_index("frequency")
    df_all_ambient["mean"] = df_all_ambient.mean(axis=1)
    df_all_ambient = df_all_ambient.reset_index()
    fig.add_scatter(x=df_all_ambient['frequency'], y=df_all_ambient['mean'], mode='lines', name="mean ambient")

    # fig.write_html("../../outputs/signal_run_1.html")
    fig.show()


if __name__ == "__main__":
    plotting_signal_raw()
