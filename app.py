import streamlit as st  
# from wasteclassification import load_images  # Import your functions from the Jupyter notebook  
import pandas as pd
import numpy as np
import time


def main():  
    st.title("My Jupyter Notebook App")  
    
    # Add your content here, for example:  
    st.write("Welcome to my Streamlit app!")  
    
    # Call your main function from the notebook  
    result = "Hello"  # load_images()
    
    # Display the result  
    st.write(result)  
    df = pd.DataFrame(np.random.randn(15, 3), columns=(["A", "B", "C"]))
    my_data_element = st.line_chart(df)

    for tick in range(10):
        time.sleep(.5)
        add_df = pd.DataFrame(np.random.randn(1, 3), columns=(["A", "B", "C"]))
        my_data_element.add_rows(add_df)

    st.button("Regenerate")
    
if __name__ == "__main__":  
    main()