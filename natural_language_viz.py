import json
import requests
import os
import pandas as pd
import numpy as np
import streamlit as st
from data_visualizer import DataVisualizer

# Add this right after your imports
# Custom JSON encoder to handle NumPy arrays and other special types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        if isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.datetime64):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

def generate_viz_response(question, api_key, visualization_data=None, insights=None, data_df=None, fig=None):
    """
    Generate a response about visualization and insights
    
    Args:
        question (str): The user's question about the visualization
        api_key (str): OpenAI API key
        visualization_data (dict): Data about the visualization (type, parameters)
        insights (str): Text insights about the visualization
        data_df (DataFrame): The data used in the visualization
        fig (plotly.graph_objects.Figure): The Plotly figure object
        
    Returns:
        str: AI-generated response about the visualization
    """
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
        
        # Additional context for specific visualization types
        if visualization_data and "viz_type" in visualization_data:
            viz_type = visualization_data.get("viz_type")
            
            if viz_type == "violin_plot":
                full_context += """
                
                ABOUT VIOLIN PLOTS:
                Violin plots show the distribution of data across different categories:
                - The width at each point shows the density of data points at that value
                - Wider sections indicate more data points at that value
                - The box plot inside shows median, quartiles, and range
                - The kernel density estimation (KDE) is the outer shape showing the probability distribution
                """
            elif viz_type == "box_plot":
                full_context += """
                
                ABOUT BOX PLOTS:
                Box plots show the statistical distribution of values:
                - The box shows the interquartile range (IQR) - middle 50% of data
                - The line in the middle of the box is the median
                - The whiskers typically extend to 1.5 * IQR
                - Points beyond the whiskers are outliers
                """
            elif viz_type == "heatmap":
                full_context += """
                
                ABOUT HEATMAPS:
                Heatmaps show relationships between two categorical variables:
                - Each cell's color represents the value at that intersection
                - Darker/more intense colors typically represent higher values
                - The color scale shows the range of values
                - Patterns and clusters indicate relationships between variables
                """
        
        # Prepare the prompt
        prompt = f"""
You are an expert data visualization assistant. I need you to answer questions about a visualization and its insights.

CONTEXT:
{full_context}

USER QUESTION:
{question}

Please provide a helpful, accurate, and concise response based on the information provided. 
The user is looking at a data visualization right now, so make your explanation specific to what they can see.
"""
        
        # Create conversation history from session state if available
        conversation_history = []
        if 'viz_chat_history' in st.session_state and len(st.session_state.viz_chat_history) > 0:
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

def create_visualization_from_text(text_prompt, data_df, api_key):
    """
    Enhanced version that generates visualization parameters based on natural language.
    Uses a more robust OpenAI prompt with better error handling and parameter mapping.
    
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

    # Get sample data and statistics to help the model understand the data better
    data_stats = {}
    for col, col_type in column_info.items():
        if col_type == "numeric":
            data_stats[col] = {
                "min": float(data_df[col].min()),
                "max": float(data_df[col].max()),
                "mean": float(data_df[col].mean()),
                "median": float(data_df[col].median()),
                "type": "numeric"
            }
        elif col_type == "categorical":
            unique_values = data_df[col].unique().tolist()
            # Only include full list if there aren't too many values
            if len(unique_values) <= 15:
                data_stats[col] = {
                    "unique_values": unique_values,
                    "count": len(unique_values),
                    "type": "categorical"
                }
            else:
                data_stats[col] = {
                    "unique_values": unique_values[:5] + ["...", unique_values[-1]],
                    "count": len(unique_values),
                    "type": "categorical"
                }
        elif col_type == "date":
            data_stats[col] = {
                "min": str(data_df[col].min()),
                "max": str(data_df[col].max()),
                "type": "date"
            }

    # Detailed descriptions of visualization types and their required parameters
    visualization_types = {
        "bar_chart": {
            "description": "Bar chart comparing values across categories",
            "required_params": ["x_column", "y_column"],
            "optional_params": ["color_column", "orientation", "title"],
            "common_use_cases": "Comparing values across different categories",
            "example": "Show Revenue by Region as a bar chart"
        },
        "line_chart": {
            "description": "Line chart showing trends over time or ordered categories",
            "required_params": ["x_column", "y_column"],
            "optional_params": ["color_column", "title"],
            "common_use_cases": "Showing trends over time, progression, or sequences",
            "example": "Display Revenue trends over time as a line chart"
        },
        "pie_chart": {
            "description": "Pie chart showing proportion of total for each category",
            "required_params": ["names_column", "values_column"],
            "optional_params": ["title"],
            "common_use_cases": "Showing composition or proportion of a total",
            "example": "Show breakdown of Revenue by Region as a pie chart"
        },
        "scatter_plot": {
            "description": "Scatter plot showing relationship between two numeric variables",
            "required_params": ["x_column", "y_column"],
            "optional_params": ["color_column", "size_column", "title"],
            "common_use_cases": "Examining correlation or relationships between numeric variables",
            "example": "Plot the relationship between Revenue and Profit"
        },
        "grouped_bar": {
            "description": "Grouped bar chart comparing values across categories with sub-groups",
            "required_params": ["x_column", "y_column", "group_column"],
            "optional_params": ["title"],
            "common_use_cases": "Comparing values across categories with an additional grouping dimension",
            "example": "Compare Revenue by Region, grouped by Product Category"
        },
        "stacked_bar": {
            "description": "Stacked bar chart showing composition within categories",
            "required_params": ["x_column", "y_column", "stack_column"],
            "optional_params": ["title"],
            "common_use_cases": "Showing both total values and composition within categories",
            "example": "Show Revenue by Region, stacked by Product Category"
        },
        "histogram": {
            "description": "Histogram showing distribution of a numeric variable",
            "required_params": ["column"],
            "optional_params": ["bins", "color_column", "title"],
            "common_use_cases": "Analyzing distribution of numeric data",
            "example": "Show distribution of Order Values"
        },
        "box_plot": {
            "description": "Box plot showing statistical distribution by categories",
            "required_params": ["x_column", "y_column"],
            "optional_params": ["color_column", "title"],
            "common_use_cases": "Comparing distributions across categories",
            "example": "Show Profit distribution by Region using box plots"
        },
        "heatmap": {
            "description": "Heatmap showing relationships between two categorical variables and a numeric value",
            "required_params": ["x_column", "y_column", "z_column"],
            "optional_params": ["title"],
            "common_use_cases": "Visualizing matrix data or correlation",
            "example": "Create a heatmap of Revenue by Region and Product Category"
        },
        "time_series": {
            "description": "Time series chart showing values over time",
            "required_params": ["date_column", "value_column"],
            "optional_params": ["color_column", "resolution", "title"],
            "common_use_cases": "Analyzing trends over time",
            "example": "Show Revenue over time as a time series"
        },
        "area_chart": {
            "description": "Area chart showing cumulative values",
            "required_params": ["x_column", "y_column"],
            "optional_params": ["group_column", "title"],
            "common_use_cases": "Showing cumulative values or stacked areas",
            "example": "Display cumulative Revenue by date as an area chart"
        },
        "bubble_chart": {
            "description": "Bubble chart showing relationship between three numeric variables",
            "required_params": ["x_column", "y_column", "size_column"],
            "optional_params": ["color_column", "title"],
            "common_use_cases": "Visualizing relationships between three variables",
            "example": "Create a bubble chart of Revenue vs Profit with Customer Count as bubble size"
        },
        "violin_plot": {
            "description": "Violin plot showing distribution density by categories",
            "required_params": ["x_column", "y_column"],
            "optional_params": ["color_column", "title"],
            "common_use_cases": "Comparing detailed distributions across categories",
            "example": "Show Revenue distribution by Region as a violin plot"
        },
        "sankey_diagram": {
            "description": "Sankey diagram showing flow between categories",
            "required_params": ["source_column", "target_column", "value_column"],
            "optional_params": ["title"],
            "common_use_cases": "Visualizing flow or transfer between categories",
            "example": "Create a Sankey diagram showing Revenue flow from Region to Product Category"
        }
    }
    
    # Create a more comprehensive prompt
    prompt = f"""
You are an expert data visualization assistant. Based on the user's natural language request, determine the most appropriate visualization type and ALL required parameters to create that visualization.

# DATA INFORMATION
The DataFrame has these columns with their types:
{json.dumps(column_info, indent=2)}

# DATA SAMPLE AND STATISTICS
Here are statistics and sample data for the columns:
{json.dumps(data_stats, indent=2)}

# VISUALIZATION TYPES AND REQUIREMENTS
Available visualization types with their required and optional parameters:
{json.dumps(visualization_types, indent=2)}

# USER'S VISUALIZATION REQUEST
"{text_prompt}"

# INSTRUCTIONS
1. Analyze the user's request and determine the most appropriate visualization type
2. Identify which columns should be used for each required parameter
3. For common analytical phrases, map them to corresponding columns (e.g. "sales" likely refers to "Revenue" column)
4. Make sure ALL required parameters for the chosen visualization type are included
5. Add a descriptive title that explains what the visualization shows
6. Add a brief description explaining why this visualization is appropriate for the request

# RESPONSE FORMAT
Return ONLY a valid JSON object with the following structure:
```json
{{
    "viz_type": "visualization_type_from_the_list",
    "description": "Brief explanation of why this visualization is appropriate",
    "parameters": {{
        // Include ALL required parameters for the selected visualization type
        // Only include parameters relevant to the chosen visualization type
        // Use actual column names from the dataset
    }}
}}
```

Be especially careful with the parameters for each visualization type. For GROUPED BAR CHARTS, you MUST include x_column, y_column, AND group_column. For PIE CHARTS, use names_column and values_column parameters.
"""

    try:
        # Make API call to OpenAI
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "gpt-4o",  # Using more capable model for better parameter mapping
            "messages": [
                {"role": "system", "content": "You are a data visualization expert that precisely converts natural language requests into visualization specifications with all required parameters."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # Lower temperature for more precise responses
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}  # Ensure we get valid JSON
        }

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
                model="gpt-4o"
            )
        
        # Parse the response content as JSON
        response_content = response_data["choices"][0]["message"]["content"]
        viz_params = json.loads(response_content)
        
        # Verify all required parameters are present for the chosen visualization type
        viz_type = viz_params.get("viz_type")
        if viz_type not in visualization_types:
            raise ValueError(f"Invalid visualization type: {viz_type}")
            
        required_params = visualization_types[viz_type]["required_params"]
        parameters = viz_params.get("parameters", {})
        
        # Check if all required parameters are present
        missing_params = [param for param in required_params if param not in parameters]
        if missing_params:
            # If parameters are missing, try to infer reasonable defaults
            if viz_type == "grouped_bar" and "group_column" in missing_params:
                # Look for categorical columns that aren't already used
                unused_categorical = [col for col, type_ in column_info.items() 
                                    if type_ == "categorical" and col not in parameters.values()]
                if unused_categorical and "x_column" in parameters:
                    # Use the first unused categorical column that isn't already the x_column
                    x_col = parameters.get("x_column")
                    available_columns = [col for col in unused_categorical if col != x_col]
                    if available_columns:
                        parameters["group_column"] = available_columns[0]
                        st.info(f"Added missing required parameter 'group_column': {available_columns[0]}")
                    else:
                        raise ValueError(f"Cannot infer missing required parameter(s): {missing_params}")
                else:
                    raise ValueError(f"Cannot infer missing required parameter(s): {missing_params}")
            else:
                raise ValueError(f"Missing required parameter(s) for {viz_type}: {missing_params}")
                
        # Return the complete visualization parameters
        return {
            "viz_type": viz_type,
            "description": viz_params.get("description", ""),
            "parameters": parameters
        }
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from OpenAI: {e}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error communicating with OpenAI API: {e}")
    except Exception as e:
        raise ValueError(f"Error generating visualization parameters: {str(e)}")


def add_nl_visualization_tab(app_tabs, analyzer, session_state):
    """Add a natural language visualization tab to the app with enhanced processing."""
    from data_visualizer import DataVisualizer

    # Create a new tab for NL visualizations
    with app_tabs["NL Visualizations"]:
        st.title("Natural Language Visualization")
        st.write("Create visualizations by describing what you want in plain English.")

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
            - "Show me a grouped bar chart of revenue by region, grouped by channel"
            """)

            viz_prompt = st.text_area("Your visualization request:",
                                    placeholder="Example: Show me the total revenue by product category as a bar chart",
                                    height=80)

            advanced_options = st.expander("Advanced Options")
            with advanced_options:
                show_debug_info = st.checkbox("Show debug information", value=False,
                                           help="Display the raw parameters used to create the visualization")
                
                # Allow selecting model for natural language processing
                nl_model = st.selectbox(
                    "AI Model for Processing",
                    ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                    index=0,
                    help="Select which model to use for natural language processing"
                )

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

                        # Show debug information if requested
                        if show_debug_info:
                            st.subheader("Debug Information")
                            st.json(viz_params)

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
                    
                    # Suggest improvements to the prompt
                    if "missing required" in str(e).lower():
                        missing_param = str(e).split(":")[-1].strip()
                        viz_type = str(e).split("for")[1].split(":")[0].strip()
                        st.info(f"Try being more specific about the {missing_param} for your {viz_type}.")
                        
                        # Provide examples based on the visualization type
                        if "grouped_bar" in viz_type.lower():
                            st.markdown("""
                            **Examples for grouped bar charts:**
                            - "Show a grouped bar chart of Revenue by Product_Category, grouped by Channel"
                            - "Create a grouped bar chart comparing Revenue across Regions, grouped by Product_Category"
                            """)
        
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
