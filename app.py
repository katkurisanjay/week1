import streamlit as st  
# from wasteclassification import load_images  # Import your functions from the Jupyter notebook  

def main():  
    st.title("My Jupyter Notebook App")  
    
    # Add your content here, for example:  
    st.write("Welcome to my Streamlit app!")  
    
    # Call your main function from the notebook  
    result = "Hello"  # load_images()
    
    # Display the result  
    st.write(result)  
    
if __name__ == "__main__":  
    main()