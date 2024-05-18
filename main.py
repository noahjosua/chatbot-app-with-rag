import subprocess


def run_streamlit():
    subprocess.run(["streamlit", "run", ".\\start_streamlit_app.py"])


if __name__ == "__main__":
    run_streamlit()
