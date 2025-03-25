import streamlit as st
import subprocess

def main():
    st.title("Check installed packages")
    cmd = subprocess.run(["pip", "show", "weaviate-client"], capture_output=True, text=True)
    st.write("### pip show weaviate-client:")
    st.write(cmd.stdout)

    # もし全パッケージを出したいなら:
    cmd2 = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
    st.write("### pip freeze:")
    st.write(cmd2.stdout)

if __name__ == "__main__":
    main()
