import streamlit as st
import pandas as pd
import os
import requests

def add_viz_chat_interface(api_key, visualization_data=None, insights=None, data_df=None):
    """
    Add a chat interface for asking questions about visualizations and insights
    
    Args:
        api_key (str): OpenAI API key
        visualization_data (dict): Data about the visualization (type, parameters)
        insights (str): Text insights about the visualization
        data_df (DataFrame): The data used in the visualization
    """
    # Initialize chat history in session state if it doesn't exist
    if 'viz_chat_history' not in st.session_state:
        st.session_state.viz_chat_history = []
    
    # Display chat header
    st.subheader("Ask About This Visualization")
    st.write("Ask questions about the visualization or data insights")
    
    # Display the chat history
    for message in st.session_state.viz_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input for new question
    user_question = st.chat_input("Ask a question about this visualization...")
    
    # Clear chat history button
    if st.session_state.viz_chat_history and st.button("Clear Chat History", key="clear_viz_chat"):
        st.session_state.viz_chat_history = []
        st.experimental_rerun()
    
    if user_question:
        # Add user question to chat history
        st.session_state.viz_chat_history.append({"role": "user", "content": user_question})
        
        # Display user question in the current session
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = generate_viz_response(user_question, api_key, visualization_data, insights, data_df)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.viz_chat_history.append({"role": "assistant", "content": response})

def generate_viz_response(question, api_key, visualization_data=None, insights=None, data_df=None):
    """Generate a response about visualization and insights"""
    try:
        # Create a context with visualization information
        viz_context = ""
        if visualization_data:
            viz_context += f"Visualization Type: {visualization_data.get('viz_type', 'unknown')}\n"
            viz_context += "Visualization Parameters:\n"
            for key, value in visualization_data.get('parameters', {}).items():
                viz_context += f"- {key}: {value}\n"
        
        # Add insights if available
        insights_context = ""
        if insights:
            insights_context = f"\nVisualization Insights:\n{insights}\n"
        
        # Add data summary if available
        data_context = ""
        if data_df is not None:
            # Get basic data stats
            data_context = f"\nData Summary:\n"
            data_context += f"- Shape: {data_df.shape[0]} rows, {data_df.shape[1]} columns\n"
            
            # Add column info
            data_context += "- Columns:\n"
            for col in data_df.columns:
                if pd.api.types.is_numeric_dtype(data_df[col]):
                    data_context += f"  - {col} (numeric): min={data_df[col].min()}, max={data_df[col].max()}, mean={data_df[col].mean():.2f}\n"
                else:
                    data_context += f"  - {col} (categorical): {data_df[col].nunique()} unique values\n"
            
            # Add a small sample
            data_context += "\nData Sample (top 3 rows):\n"
            data_context += data_df.head(3).to_string()
        
        # Combine all context information
        full_context = viz_context + insights_context + data_context
        
        # Prepare the prompt
        prompt = f"""
You are an expert data visualization assistant. I need you to answer questions about a visualization and its insights.

CONTEXT:
{full_context}

USER QUESTION:
{question}

Please provide a helpful, accurate, and concise response based only on the information provided. 
If you don't have enough information to answer the question, please say so clearly.
"""
        
        # Create conversation history from session state
        conversation_history = []
        if len(st.session_state.viz_chat_history) > 0:
            for msg in st.session_state.viz_chat_history[-6:]:  # Last 3 exchanges (up to 6 messages)
                conversation_history.append({"role": msg["role"], "content": msg["content"]})
                
        # Call OpenAI API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a data visualization expert that helps users understand visualizations and data insights."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        # Add conversation history if it exists
        if conversation_history:
            # Insert history before the final prompt
            payload["messages"] = [payload["messages"][0]] + conversation_history + [payload["messages"][1]]
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload
        )
        response.raise_for_status()
        response_data = response.json()
        
        # Track token usage if session state has token tracker
        if hasattr(st.session_state, 'token_tracker'):
            st.session_state.token_tracker.track_api_call(
                prompt=prompt,
                response=response_data,
                model="gpt-3.5-turbo"
            )
        
        return response_data["choices"][0]["message"]["content"]
        
    except Exception as e:
        return f"I'm sorry, I encountered an error while generating a response: {str(e)}"
