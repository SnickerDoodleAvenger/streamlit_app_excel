import json
import requests
import os
import pandas as pd
import streamlit as st
from data_visualizer import DataVisualizer
from viz_chat_interface import generate_viz_response


def create_visualization_from_text(text_prompt, data_df, api_key):
    """
    Generates visualization parameters based on a natural language description.

    Args:
        text_prompt (str): Natural language description of the desired visualization
        data_df (pd.DataFrame): The dataframe containing the data
        api_key (str): OpenAI API key

    Returns:
        dict: Visualization parameters that can be passed to the DataVisualizer
    """
    # Get column information to help the model understand the data
    column_info = {}
    for col in data_df.columns:
        if pd.api.types.is_numeric_dtype(data_df[col]):
            column_info[str(col)] = "numeric"
        elif pd.api.types.is_datetime64_dtype(data_df[col]):
            column_info[str(col)] = "date"
        else:
            column_info[str(col)] = "categorical"

    # Prepare the prompt for the OpenAI API
    column_descriptions = "\n".join([f"- {col}: {type_}" for col, type_ in column_info.items()])
    unique_values = {col: data_df[col].nunique() for col in data_df.columns
                     if column_info[str(col)] == "categorical" and data_df[col].nunique() < 15}
    unique_values_str = "\n".join([f"- {col}: {data_df[col].unique().tolist()}" for col in unique_values])

    # Map of common chart types for better inference
    chart_types = {
        "bar_chart": "Bar chart comparing categories",
        "line_chart": "Line chart showing trends over values or time",
        "scatter_plot": "Scatter plot showing relationship between two numeric variables",
        "pie_chart": "Pie chart showing composition/distribution",
        "histogram": "Histogram showing distribution of one numeric variable",
        "box_plot": "Box plot showing statistical distribution by categories",
        "heatmap": "Heatmap showing relationship between two categorical variables and a numeric value",
        "time_series": "Time series plot for data over time",
        "grouped_bar": "Grouped bar chart comparing multiple categories",
        "stacked_bar": "Stacked bar chart showing composition within categories",
        "area_chart": "Area chart showing cumulative values",
        "bubble_chart": "Bubble chart showing relationship with size representing a third variable",
        "violin_plot": "Violin plot showing distribution density by categories",
        "sankey_diagram": "Sankey diagram showing flow between two categorical variables"
    }
    charts_description = "\n".join([f"- {key}: {desc}" for key, desc in chart_types.items()])

    prompt = f"""
You are an expert data visualization assistant. Based on the user's request, determine the appropriate visualization type and parameters to visualize their data.

DATA INFORMATION:
The DataFrame has these columns with their types:
{column_descriptions}

For categorical columns with limited unique values:
{unique_values_str}

AVAILABLE VISUALIZATION TYPES:
{charts_description}

USER REQUEST:
"{text_prompt}"

For pie charts, use parameters 'names_column' for categories and 'values_column' for values.
For bar charts, use 'x_column' for categories and 'y_column' for values.

Return a JSON object with the following structure:
{{
    "viz_type": "The chart type (one of the keys above)",
    "description": "Brief explanation of why this visualization type is appropriate",
    "parameters": {{
        // All parameters needed for this visualization type
        // Such as x_column, y_column, color_column, etc.
        // Only include relevant parameters for the selected viz_type
    }}
}}

Make sure to only select columns that exist in the dataset and match the required data types for the chosen visualization.
"""

    try:
        # Make API call to OpenAI
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system",
                 "content": "You are a data visualization expert that converts natural language requests into visualization specifications."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        # Extract the response content
        response_data = response.json()

        # Track token usage if session state has token tracker
        if hasattr(st.session_state, 'token_tracker'):
            st.session_state.token_tracker.track_api_call(
                prompt=prompt,
                response=response_data,
                model="gpt-3.5-turbo"
            )

        response_text = response_data["choices"][0]["message"]["content"]

        # Extract the JSON part in the response (in case there's additional text)
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            try:
                viz_params = json.loads(json_str)

                # Map parameter names to match DataVisualizer's expected parameters
                if "parameters" in viz_params:
                    params = viz_params["parameters"]

                    # For pie charts
                    if viz_params["viz_type"] == "pie_chart":
                        # Rename common alternatives to the expected parameter names
                        param_mapping = {
                            "category_column": "names_column",
                            "categories": "names_column",
                            "labels": "names_column",
                            "value_column": "values_column",
                            "values": "values_column"
                        }

                        # Apply the mapping
                        for old_param, new_param in param_mapping.items():
                            if old_param in params and new_param not in params:
                                params[new_param] = params.pop(old_param)

                    # For bar charts
                    if viz_params["viz_type"] == "bar_chart":
                        param_mapping = {
                            "category_column": "x_column",
                            "categories": "x_column",
                            "value_column": "y_column",
                            "values": "y_column"
                        }

                        for old_param, new_param in param_mapping.items():
                            if old_param in params and new_param not in params:
                                params[new_param] = params.pop(old_param)

                return viz_params
            except json.JSONDecodeError:
                # Try to clean up common JSON formatting issues
                # Replace single quotes with double quotes
                json_str = json_str.replace("'", "\"")
                # Handle trailing commas
                json_str = json_str.replace(",\n}", "\n}")
                json_str = json_str.replace(",}", "}")
                try:
                    viz_params = json.loads(json_str)

                    # Apply parameter mapping after parsing
                    if "parameters" in viz_params:
                        params = viz_params["parameters"]

                        # For pie charts
                        if viz_params["viz_type"] == "pie_chart":
                            param_mapping = {
                                "category_column": "names_column",
                                "categories": "names_column",
                                "labels": "names_column",
                                "value_column": "values_column",
                                "values": "values_column"
                            }

                            for old_param, new_param in param_mapping.items():
                                if old_param in params and new_param not in params:
                                    params[new_param] = params.pop(old_param)

                        # For bar charts
                        if viz_params["viz_type"] == "bar_chart":
                            param_mapping = {
                                "category_column": "x_column",
                                "categories": "x_column",
                                "value_column": "y_column",
                                "values": "y_column"
                            }

                            for old_param, new_param in param_mapping.items():
                                if old_param in params and new_param not in params:
                                    params[new_param] = params.pop(old_param)

                    return viz_params
                except:
                    raise ValueError("Could not parse the response as JSON")
        else:
            raise ValueError("Could not find JSON content in the response")

    except Exception as e:
        st.error(f"Error generating visualization parameters: {str(e)}")
        raise


def add_nl_visualization_tab(app_tabs, analyzer, session_state):
    """Add a natural language visualization tab to the app."""
    from data_visualizer import DataVisualizer

    # Create a new tab for NL visualizations
    with app_tabs["NL Visualizations"]:
        st.title("Natural Language Visualization")
        st.write("Create visualizations by describing what you want and chat with the AI about insights.")

        # Initialize DataVisualizer if needed
        if 'visualizer' not in session_state or session_state.visualizer is None:
            if session_state.filtered_data is not None:
                session_state.visualizer = DataVisualizer(session_state.filtered_data)
                data_for_viz = session_state.filtered_data
            elif session_state.excel_data is not None:
                session_state.visualizer = DataVisualizer(session_state.excel_data)
                data_for_viz = session_state.excel_data
            else:
                st.warning("No data available for visualization. Please upload an Excel file first.")
                return
        else:
            # Update the visualizer if data has changed
            if session_state.filters_applied and session_state.filtered_data is not None:
                session_state.visualizer = DataVisualizer(session_state.filtered_data)
                data_for_viz = session_state.filtered_data
            else:
                session_state.visualizer = DataVisualizer(session_state.excel_data)
                data_for_viz = session_state.excel_data

        # Get API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = st.text_input("OpenAI API Key:", type="password",
                                    help="Enter your OpenAI API key to enable natural language visualization")
        
        # Create two main containers - one for viz generation, one for persistent viz display
        viz_input = st.container()
        viz_display = st.container()
        chat_section = st.container()
        
        # First container: Visualization request input 
        with viz_input:
            st.subheader("Describe the Visualization You Want")
            st.markdown("""
            Examples:
            - "Show me revenue by region as a bar chart"
            - "Create a line chart of revenue over time"
            - "Compare sales between product categories and channels"
            - "Create a violin plot of revenue by product category to see the distribution"
            """)

            viz_prompt = st.text_area("Your visualization request:",
                                    placeholder="Example: Show me the total revenue by product category as a bar chart",
                                    height=80)

            if st.button("Generate Visualization", key="nl_viz_generate_button") and viz_prompt and api_key:
                try:
                    with st.spinner("Creating your visualization..."):
                        # Generate visualization parameters from the text description
                        viz_params = create_visualization_from_text(viz_prompt, data_for_viz, api_key)

                        # Store the visualization data in session state for the chat interface
                        session_state.current_viz_data = viz_params

                        # Get the parameters for the visualization
                        viz_type = viz_params.get("viz_type")
                        params = viz_params.get("parameters", {})

                        # Add viz_type to the parameters
                        params["viz_type"] = viz_type

                        # Create the visualization
                        fig = session_state.visualizer.create_visualization(**params)
                        
                        # Store the figure in session state for the chat interface
                        session_state.current_fig = fig

                        # Generate insights about the visualization
                        insights = session_state.visualizer.generate_visualization_insights(
                            fig, 
                            viz_type, 
                            params, 
                            data_for_viz,
                            api_key
                        )
                        
                        # Store insights in session state for the chat interface
                        session_state.current_viz_insights = insights
                        
                        # Store the parameters for reference
                        session_state.current_viz_params = params
                        
                        # Store the interpretation for reference
                        session_state.current_viz_interpretation = viz_params.get("description", "")
                        
                        # Set a flag to indicate we have a visualization
                        session_state.has_visualization = True
                
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
                    st.write("Please try a different description or be more specific about what you'd like to visualize.")
        
        # Second container: Always display visualization if available
        with viz_display:
            if session_state.get('has_visualization', False):
                # Display interpretation
                if 'current_viz_interpretation' in session_state:
                    st.subheader("Visualization Interpretation")
                    st.write(session_state.current_viz_interpretation)
                
                # Display the visualization
                st.subheader("Visualization")
                st.plotly_chart(session_state.current_fig, use_container_width=True)
                
                # Download options
                st.download_button(
                    label="Download Visualization as HTML",
                    data=session_state.current_fig.to_html(),
                    file_name=f"{session_state.current_viz_data.get('viz_type', 'visualization')}.html",
                    mime="text/html",
                    key="download_nl_viz_html"
                )
                
                # Show used parameters for advanced users
                with st.expander("View Visualization Parameters"):
                    st.json(session_state.current_viz_params)
                
                # Display insights
                st.subheader("Visualization Insights")
                st.markdown(session_state.current_viz_insights)
                
                # Download insights
                st.download_button(
                    label="Download Insights",
                    data=session_state.current_viz_insights,
                    file_name=f"{session_state.current_viz_data.get('viz_type', 'visualization')}_insights.md",
                    mime="text/markdown",
                    key="download_nl_viz_insights"
                )
        
        # Third container: Chat interface
        with chat_section:
            # Add a divider before the chat section
            st.divider()
            
            # Add the chat interface if we have a visualization
            if session_state.get('has_visualization', False) and api_key:
                st.subheader("Ask Questions About This Visualization")
                
                # Initialize chat history in session state if it doesn't exist
                if 'viz_chat_history' not in st.session_state:
                    st.session_state.viz_chat_history = []
                
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
                            response = generate_viz_response(
                                user_question, 
                                api_key, 
                                session_state.current_viz_data, 
                                session_state.current_viz_insights, 
                                data_for_viz,
                                session_state.get('current_fig', None)
                            )
                            st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.viz_chat_history.append({"role": "assistant", "content": response})

            else:
                if not session_state.get('has_visualization', False):
                    st.info("Generate a visualization first to enable the chat interface.")
