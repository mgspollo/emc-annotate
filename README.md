## EMC Annotate

The aim of this repository is to automatically identify features in the spectrum

Deliverable 1: Automate the pattern finding for power supplies, HDMI and displayport.  Alex has been able to visually see patterns that define these.  So using an appropriate pattern finding model be able to take in a new input spectrum and detect whether it has the pattern for power supply, HDMI or displayport with a strength score.

Deliverable 2: Display the score/flags visually on the viewer so a new file can be viewed with a score or a suggestion associated with them

Deliverable 3: Allow the viewer and associated pattern model to be updated to take in more patterns by Kenlock and provide documentation on the requirements for the file inputs etc.

### Data setup

Currently in the import_data.py the data folder is set-up like following:

data/test_data/
- run1/
- run2/
- run3/
  - 10001.csv
  - 10001 - Ambient.csv
  - 10002.csv
  - 10002 - Ambient.csv
  - etc.
- run1_metadata.xlsx
- run2_metadata.xlsx
- run3_metadata.xlsx

New files should be appended to the end - and new runs should be put in a new folder with a new download of the Schema.xlsx for backwards compatibility

These files are then brought together to the processed_data folder where

data/processed_data/
- all_tests.csv
- metadata.csv