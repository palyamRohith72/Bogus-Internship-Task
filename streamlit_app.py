import streamlit as st
import pandas as pd
from groq import Groq
import os

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("hotels_data_preprocessed.csv")

# Initialize Groq client
def get_groq_client(api_key):
    return Groq(api_key=api_key)

# Function to query Groq with complete dataset
def query_complete_dataset(client, query, df):
    # Convert dataframe to string for context
    data_context = f"""
    Here is the dataset information:
    - Number of rows: {df.shape[0]}
    - Number of columns: {df.shape[1]}
    - Columns: {', '.join(df.columns)}
    - Sample data: {df.head().to_dict()}
    
    The user's query is: {query}
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful data analyst assistant. You have access to hotel bookings data. Provide detailed, accurate responses to user queries about the data."
                },
                {
                    "role": "user",
                    "content": data_context
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=4096
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error querying Groq API: {str(e)}"

# Function to query Groq with subset of data
def query_data_subset(client, query, df_subset):
    # Convert subset dataframe to string for context
    data_context = f"""
    Here is the subset of data you're working with:
    - Number of rows: {df_subset.shape[0]}
    - Columns: {', '.join(df_subset.columns)}
    - Data: {df_subset.to_dict()}
    
    The user's query is: {query}
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful data analyst assistant. You have access to a subset of hotel bookings data. Provide detailed, accurate responses to user queries about this specific data subset."
                },
                {
                    "role": "user",
                    "content": data_context
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.3,
            max_tokens=4096
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error querying Groq API: {str(e)}"

def main():
    st.set_page_config(page_title="Hotel Bookings Analysis", layout="wide")
    
    # Load data
    df = load_data()
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Groq API key input
    st.sidebar.info("Go to Groq Cloud (https://console.groq.com/) to generate a free API key and provide it below")
    api_key = st.sidebar.text_input("Groq API Key", type="password")
    
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar to proceed")
        return
    
    # Initialize Groq client
    try:
        client = get_groq_client(api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {str(e)}")
        return
    
    # Search mode selection
    search_mode = st.sidebar.radio(
        "Search Mode",
        options=["Search on complete dataset", "Search on sub data"],
        index=0
    )
    
    # Main content based on search mode
    if search_mode == "Search on complete dataset":
        st.title("Hotel Bookings Analysis - Complete Dataset")
        
        query = st.sidebar.text_area(
            "Enter your query about the complete dataset:",
            height=150,
            placeholder="E.g., What are the trends in hotel bookings by month?"
        )
        
        if st.sidebar.button("Trigger Action", key="complete_dataset"):
            with st.spinner("Processing your query..."):
                response = query_complete_dataset(client, query, df)
                st.markdown("### Query Result")
                st.markdown(response)
                
                # Show some basic info about the dataset
                st.markdown("### Dataset Overview")
                st.write(f"Total rows: {df.shape[0]}")
                st.write(f"Total columns: {df.shape[1]}")
                st.write("First 5 rows:")
                st.dataframe(df.head())
    
    else:  # Search on sub data
        st.title("Hotel Bookings Analysis - Subset of Data")
        
        # Column selection
        selected_columns = st.sidebar.multiselect(
            "Select columns to analyze:",
            options=df.columns,
            default=["hotel", "arrival_date_month", "adults", "children", "country"]
        )
        
        # Row range selection
        col1, col2 = st.sidebar.columns(2)
        start_idx = col1.number_input(
            "Start index:",
            min_value=0,
            max_value=df.shape[0]-1,
            value=0,
            step=1
        )
        end_idx = col2.number_input(
            "End index:",
            min_value=0,
            max_value=df.shape[0]-1,
            value=min(100, df.shape[0]-1),
            step=1
        )
        
        # Ensure end index is greater than start index
        if end_idx <= start_idx:
            st.sidebar.warning("End index must be greater than start index")
            return
        
        query = st.sidebar.text_area(
            "Enter your query about the selected data subset:",
            height=150,
            placeholder="E.g., What's the distribution of adults in this subset?"
        )
        
        if st.sidebar.button("Trigger Action", key="sub_data"):
            with st.spinner("Processing your query..."):
                # Create subset
                df_subset = df[selected_columns].iloc[start_idx:end_idx+1]
                
                # Show the subset
                st.markdown("### Selected Data Subset")
                st.write(f"Showing rows {start_idx} to {end_idx} of selected columns")
                st.dataframe(df_subset)
                
                if query:
                    response = query_data_subset(client, query, df_subset)
                    st.markdown("### Query Result")
                    st.markdown(response)

if __name__ == "__main__":
    main()
