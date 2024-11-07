import pandas as pd
import csv

defined_thresholds = {
    "EC100": {"start": 240000000, "stop": 750000000, "level": -70},
    "EC150": {"start": 750000000, "stop": 950000000, "level": -65},
}


def read_practice_data():
    """
    Function to read in the practice data, now redundant as the real test data has been produced.
    :return:
    """
    df_all = pd.DataFrame()
    for i, file in enumerate(["1", "2", "3"]):
        df = pd.read_csv(filepath_or_buffer="../../data/practice_data/dataoutput" + file + ".csv", header=None,
                         names=["frequency", file])
        if i == 0:
            df_all = df
        else:
            df_all = df_all.merge(df, on="frequency")
    return df_all


def find_header_row(file, header_name="XUnits"):
    """
    Generic function that finds the first occurence of header_name in the csv provided and outputs the **next** row
    Works as an input to 'skiprows' in read_csv

    :param file: file to read
    :param header_name: name to find first occurrence of
    :return: the index of the next row (which should contain the data)
    """

    def get_first_element(lst):
        if lst:
            return lst[0]
        return None

    with open(file) as fin:
        reader = csv.reader(fin)
        idx = next(idx for idx, row in enumerate(reader) if get_first_element(row) == header_name)
    return idx + 1


def generate_file_name(test_id, ambient=True):
    """
    Generates the file name for the spectrum files

    :param i: index of the test
    :param ambient: whether or not the spectrum is ambient
    :return: string of the file name
    """
    test_data_path = "../../data/test_data/"
    if ambient:
        file_name = test_data_path + "run3/" + str(test_id) + " - Ambient.csv"
    else:
        file_name = test_data_path + "run3/" + str(test_id) + ".csv"
    return file_name


def generate_column_name(test_id, ambient=True):
    """
    Generates the column name for the spectrum files

    :param test_id: index of the test
    :param ambient: whether or not the spectrum is ambient
    :return: string of the column name
    """
    if ambient:
        file_name = str(test_id) + "a"
    else:
        file_name = str(test_id)
    return file_name


def read_test_metadata(max_test_id):
    """
    Reading the metadata of the test from the metadata file

    Reads two sheets:
    - Data Table (aka fact table)
    - Data Dictionary (aka dimension table)

    These dataframes are merged together. There is some nuance as the cells in the Data Table are merged at the top
    and the Data Dictionary is transposed with respect to the Data Table

    The resulting data frame contains only the processed column names as headers and only the real, processed metadata
    as rows
    :return: df_fact_dim, dataframe
    """
    test_data_path = "../../data/test_data/"
    df_fact = pd.read_excel(test_data_path + "run3_metadata.xlsx", sheet_name="Data Table", header=[0, 1])
    df_dim_prod = pd.read_excel(test_data_path + "run3_metadata.xlsx", sheet_name="Fact Table",
                                header=[0, 1])
    df_fact = convert_column_names(df_fact)
    df_dim_prod = convert_column_names(df_dim_prod)

    df_fact_dim = pd.merge(df_fact, df_dim_prod, on='product_id')
    df_fact_dim = df_fact_dim.dropna(subset=["date"]).dropna(how='all', axis=1)
    df_fact_dim = df_fact_dim[df_fact_dim.test_id < max_test_id]
    df_fact_dim = df_fact_dim.loc[:, df_fact_dim.any()]
    return df_fact_dim


def convert_column_names(df):
    """
    Convert the column names that are in the data table into those allowed by the data dictionary
    :param df:
    :return:
    """
    test_data_path = "../../data/test_data/"
    df = df.T.reset_index()
    df_dict = pd.read_excel(test_data_path + "run3_metadata.xlsx", sheet_name="Data Dictionary", header=0,
                            index_col=[0, 1]).reset_index()
    df_dict = df_dict.rename(columns={"Our Name": "level_0"})
    df_cleaned = pd.merge(df_dict, df, on=["level_0", "level_1"], how='outer').drop(
        columns=["level_0", "level_1", "Type", "Description", "Allowed Values", "Datatype", "Notes"])
    df_cleaned = df_cleaned.set_index("Data Name").T
    df_cleaned = df_cleaned.dropna(how='all', axis=1)
    if "test_id" in df_cleaned.columns:
        df_cleaned = df_cleaned.dropna(subset=["test_id"])
        df_cleaned['test_id'] = df_cleaned['test_id'].astype(int)

    if "product_id" in df_cleaned.columns:
        df_cleaned = df_cleaned.dropna(subset=["product_id"])
        df_cleaned['product_id'] = df_cleaned['product_id'].astype(int).astype(str).str.zfill(4)
    return df_cleaned


def read_test_data(max_test_id):
    """
    Function that reads in the spectrums

    This function calls a clever helper function find_header_row as it was found that the number of rows to skip is not
    always constant. This function finds the first instance of the variable header_name and outputs the index of that,
    to ensure that the provided dataframe for the spectrum contains only the spectrum and not the preceeding metadata.
    This number is normally about 352, varying between 348 and 356.

    :param max_i: Shouldn't be needed as the max_i should be equal to the number of tests in the metadata, inevitably
    this isnt' the case so needs to be a separate parameter.
    :return: df_fact_dim, a dataframe containing all the metadata and then the spectrums as the last two columns,
    which are themselves dataframes nested within the dataframes
    """
    df_all_signal = pd.DataFrame()
    for i, test_id in enumerate(range(10001, max_test_id)):
        for is_ambient in [True, False]:
            df_signal = pd.read_csv(generate_file_name(test_id, is_ambient),
                                    skiprows=find_header_row(generate_file_name(test_id, is_ambient)),
                                    nrows=32000, usecols=[0, 1], names=["frequency", "intensity"])
            df_signal = df_signal.apply(pd.to_numeric)
            df_signal = df_signal.drop_duplicates(subset='frequency')
            if i == 0 and is_ambient:
                df_all_signal = df_signal.rename(columns={"intensity": generate_column_name(test_id, is_ambient)})
            else:
                df_all_signal = pd.merge(df_all_signal, df_signal.rename(columns={"intensity": generate_column_name(
                    test_id, is_ambient)}),
                                         on='frequency')
    df_all_signal = df_all_signal.set_index('frequency')
    return df_all_signal


def normalise_spectra(max_test_id):
    df = read_test_data(max_test_id=10080)
    for i, test_id in enumerate(range(10001, max_test_id)):
        df[str(test_id) + "n"] = df[str(test_id)] - df[str(test_id) + "a"]
        # df[str(test_id) + "n"] = (df[str(test_id) + "n"] - min(df[str(test_id) + "n"])
    save_tests(df)
    return df

def filter_only_tests(max_test_id):
    df = normalise_spectra(max_test_id)
    df = df[[str(i) + "n" for i in range(10001, max_test_id) if not i == 10065]]
    return df

def process_metadata(max_test_id):
    df_metadata = read_test_metadata(max_test_id)
    df_metadata["is_protocol_constant"] = df_metadata["display_output_protocol"] == df_metadata["display_receive_protocol"]
    df_metadata = df_metadata[[
        "is_protocol_constant", 'is_power_supply_present', 'power_supply_loading', 'test_id', 'product_description', 'display_output_protocol', 'display_receive_protocol',
    ]]
    df_metadata['test_id'] = df_metadata['test_id'].astype(str)
    save_metadata(df_metadata)
    return df_metadata.set_index('test_id')


def save_metadata(df_metadata):
    df_metadata.to_csv("../../data/processed_data/metadata.csv", index=False)


def save_tests(df):
    df.to_csv("../../data/processed_data/all_tests.csv")


if __name__ == "__main__":
    max_test_id = 10080
    df_all_signal = read_test_data(max_test_id)
    df_metadata = process_metadata(max_test_id)
