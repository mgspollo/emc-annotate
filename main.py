import argparse

def main():
    from src.visualisation.dash_app import app
    app.run_server(debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data analysis pipeline.")
    main()