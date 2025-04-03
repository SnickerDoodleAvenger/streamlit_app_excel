import streamlit as st
import pandas as pd
import openai
import time
from data_visualizer import DataVisualizer  # Import your DataVisualizer class

def setup_chat_ui():
    """
    Setup and handle the chat interface for the Ask Questions tab
    """
    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat header
    st.header("Chat with Your Data")
    st.write("Ask questions about your data and get AI-powered responses.")
    
    # Display the chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input for new question
    user_question = st.chat_input("Ask a question about your data...")
    
    if user_question:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display user question
        with st.chat_message("user"):
            st.write(user_question)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(user_question, st.session_state.df)
                st.write(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

def generate_response(question, dataframe):
    """
    Generate a response to the user's question about the data
    
    Args:
        question: The user's question
        dataframe: The dataframe containing the data
    
    Returns:
        str: The response to the user's question
    """
    # Check if OpenAI API key is available
    api_key = st.session_state.get('openai_api_key', None)
    
    if api_key:
        try:
            # Use OpenAI to generate a response
            return generate_openai_response(question, dataframe, api_key)
        except Exception as e:
            st.error(f"Error generating OpenAI response: {e}")
            # Fall back to basic response if OpenAI fails
            return generate_basic_response(question, dataframe)
    else:
        # Generate a basic response without OpenAI
        return generate_basic_response(question, dataframe)

def generate_openai_response(question, dataframe, api_key):
    """Generate response using OpenAI API"""
    # Set the API key
    openai.api_key = api_key
    
    # Prepare data summary for context
    data_summary = prepare_data_summary(dataframe)
    
    # Prepare the prompt
    prompt = f"""
You are a data analysis assistant. Answer the following question about this dataset.

QUESTION:
{question}

DATASET SUMMARY:
{data_summary}

Provide a helpful, accurate, and concise response based on the data.
"""
    
    # Make API call
    client = openai.Client(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a data analysis assistant that helps users understand their data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    # Track token usage if token tracker exists
    if hasattr(st.session_state, 'token_tracker'):
        st.session_state.token_tracker.track_api_call(
            prompt=prompt,
            response=response,
            model="gpt-3.5-turbo"
        )
    
    # Return the response content
    return response.choices[0].message.content

def generate_basic_response(question, dataframe):
    """Generate a basic response without using OpenAI"""
    # Get basic statistics
    num_rows, num_cols = dataframe.shape
    column_names = dataframe.columns.tolist()
    
    # Check for keywords in the question to determine response type
    if any(word in question.lower() for word in ["shape", "size", "dimensions", "rows", "columns"]):
        return f"The dataset has {num_rows} rows and {num_cols} columns."
    
    elif any(word in question.lower() for word in ["column", "field", "variable"]):
        return f"The dataset contains these columns: {', '.join(column_names)}"
    
    elif any(word in question.lower() for word in ["average", "mean", "median", "min", "max", "sum"]):
        # Try to identify which column the user is asking about
        for col in dataframe.columns:
            if col.lower() in question.lower():
                if pd.api.types.is_numeric_dtype(dataframe[col]):
                    stats = dataframe[col].describe()
                    return f"Statistics for {col}:\n- Mean: {stats['mean']:.2f}\n- Median: {stats['50%']:.2f}\n- Min: {stats['min']:.2f}\n- Max: {stats['max']:.2f}"
                else:
                    return f"{col} is not a numeric column, so statistical calculations are limited."
        
        # If no specific column identified, provide summary of all numeric columns
        numeric_cols = dataframe.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats_summary = "Summary statistics for numeric columns:\n"
            for col in numeric_cols[:3]:  # Limit to first 3 to keep it manageable
                mean_val = dataframe[col].mean()
                stats_summary += f"- {col}: average = {mean_val:.2f}\n"
            if len(numeric_cols) > 3:
                stats_summary += f"(plus {len(numeric_cols) - 3} more numeric columns)"
            return stats_summary
    
    elif "correlation" in question.lower() or "related" in question.lower():
        numeric_cols = dataframe.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            corr = dataframe[col1].corr(dataframe[col2])
            return f"The correlation between {col1} and {col2} is {corr:.2f}."
    
    elif "missing" in question.lower() or "null" in question.lower():
        missing_count = dataframe.isna().sum().sum()
        return f"There are {missing_count} missing values in the dataset."
    
    # Default response for other questions
    return "I can help answer questions about your data's shape, columns, basic statistics, correlations, and missing values. Please specify what you'd like to know."

def prepare_data_summary(dataframe):
    """
    Prepare a summary of the dataframe for the OpenAI prompt
    """
    # Get basic information
    num_rows, num_cols = dataframe.shape
    dtypes = dataframe.dtypes.value_counts().to_dict()
    dtype_summary = ", ".join([f"{count} {dtype} columns" for dtype, count in dtypes.items()])
    
    # Column information
    column_info = []
    for col in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            col_summary = f"- {col} (numeric): min={dataframe[col].min()}, max={dataframe[col].max()}, mean={dataframe[col].mean():.2f}"
        elif pd.api.types.is_datetime64_dtype(dataframe[col]):
            col_summary = f"- {col} (date): range from {dataframe[col].min()} to {dataframe[col].max()}"
        else:
            nunique = dataframe[col].nunique()
            col_summary = f"- {col} (categorical): {nunique} unique values"
            if nunique <= 5:  # Only show all values if there are few of them
                unique_vals = dataframe[col].dropna().unique()
                col_summary += f" - {', '.join(str(val) for val in unique_vals)}"
        column_info.append(col_summary)
    
    # Create the summary
    summary = f"""
This dataset contains {num_rows} rows and {num_cols} columns ({dtype_summary}).

Column details:
{chr(10).join(column_info[:10])}  # Limit to 10 columns to avoid very long prompts
"""
    
    if len(column_info) > 10:
        summary += f"\n(Plus {len(column_info) - 10} more columns)"
    
    # Add sample data
    summary += "\n\nFirst 5 rows of data:\n"
    summary += dataframe.head(5).to_string()
    
    return summary

# This function should be called from your main Streamlit app
def add_chat_to_ask_questions_tab():
    setup_chat_ui()
