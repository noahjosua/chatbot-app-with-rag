import subprocess


def run_streamlit():
    subprocess.run(['streamlit', 'run', '.\\01_main.py'])


if __name__ == "__main__":
    run_streamlit()
