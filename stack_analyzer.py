# Coomment local_idx print

# python -m venv .venv
# source .venv/bin/activate.fish
# pip install streamlit
# pip install pandas
# streamlit run stack_analyzer.py

# nvcc --extended-lambda -lineinfo -res-usage -std=c++17 -arch=sm_86 -rdc=true -lineinfo -res-usage Source_12_Read_File_Dynamic_Paralleism.cu -o a.out && ./a.out > Source_12.txt

import streamlit as st
import pandas as pd
import re

def parse_output(file_content):
    stack_section = False
    ops_section = False
    valid_stack = []
    push_success = 0
    pop_success = 0
    
    for line in file_content.split('\n'):
        if "Stack after Operations" in line:
            stack_section = True
            continue
        if "Results of push" in line:
            stack_section = False
            ops_section = True
            continue
            
        if stack_section:
            if match := re.match(r"Index (\d+): (-?\d+)", line):
                index, val = int(match.group(1)), int(match.group(2))
                if val > 0:  # Valid stack entries
                    valid_stack.append((index, val))
        
        if ops_section:
            if "Push(" in line and "OK" in line:
                push_success += 1
            elif "Pop() -> (" in line:
                # Check if the line does NOT contain "FAILED" AND the result is not 0
                if "FAILED" not in line:
                    match = re.search(r"Pop\(\) -> \(([^)]+)\)", line)
                    if match:
                        result = match.group(1).strip()
                        if result != "0":
                            pop_success += 1
                
    return push_success, pop_success, valid_stack

st.title("CUDA Dynamic Parallelism Stack Operation Analyzer")
uploaded_file = st.file_uploader("Upload Output file", type="txt")

if uploaded_file:
    content = uploaded_file.getvalue().decode()
    pushes, pops, stack_data = parse_output(content)
    
    col1, col2 = st.columns(2)
    col1.metric("Successful Pushes", pushes)
    col2.metric("Successful Pops", pops)
    
    st.subheader("Valid Stack Contents")
    if stack_data:
        st.dataframe(
            pd.DataFrame(stack_data, columns=["Index", "Value"]),
            height=300,
            use_container_width=True
        )
    else:
        st.warning("No valid stack entries found")
