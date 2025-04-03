import streamlit as st
import pandas as pd
import os
import requests
import json
import numpy as np

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

def add_viz_chat_interface(api_key, visualization_data=None, insights=None, data_df=None, fig=None):
    """
    Add a chat interface for asking questions about visualizations and insights
    
    Args:
        api_key (str): OpenAI API key
        visualization_data (dict): Data about the visualization (type, parameters)
        insights (str): Text insights about the visualization
        data_df (DataFrame): The data used in the visualization
        fig (plotly.graph_objects.Figure): The Plotly figure object
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
                response = generate_viz_response(user_question, api_key, visualization_data, insights, data_df, fig)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.viz_chat_history.append({"role": "assistant", "content": response})

def extract_figure_data(fig):
    """Extract detailed information from a Plotly figure object"""
    figure_data = {}
    
    try:
        if fig is None:
            return "No figure data available"
        
        # Extract figure type
        figure_data["chart_type"] = fig._layout_obj.title.text.split(" of ")[0] if hasattr(fig, '_layout_obj') and hasattr(fig._layout_obj, 'title') else "Unknown chart type"
        
        # Extract data traces
        if hasattr(fig, 'data') and fig.data:
            traces = []
            for i, trace in enumerate(fig.data):
                trace_info = {
                    "trace_type": trace.type if hasattr(trace, 'type') else "unknown",
                    "name": trace.name if hasattr(trace, 'name') else f"Trace {i}",
                }
                
                # Handle x and y data safely (without including raw data)
                if hasattr(trace, 'x') and trace.x is not None:
                    # Just mention that x data exists but don't include values
                    trace_info["has_x_data"] = True
                    # Get length safely
                    try:
                        trace_info["x_length"] = len(trace.x)
                    except:
                        trace_info["x_length"] = "unknown"
                
                if hasattr(trace, 'y') and trace.y is not None:
                    # Just mention that y data exists but don't include values
                    trace_info["has_y_data"] = True
                    # Get length safely
                    try:
                        trace_info["y_length"] = len(trace.y)
                    except:
                        trace_info["y_length"] = "unknown"
                
                # Extract violin-specific data
                if trace.type == 'violin':
                    trace_info["kernel_density_visible"] = True
                    if hasattr(trace, 'points') and trace.points:
                        trace_info["shows_points"] = True
                        trace_info["point_type"] = trace.points
                    if hasattr(trace, 'meanline') and trace.meanline and trace.meanline.visible:
                        trace_info["shows_mean"] = True
                    if hasattr(trace, 'box') and trace.box and trace.box.visible:
                        trace_info["shows_boxplot"] = True

                # Extract bar chart specific data
                elif trace.type == 'bar':
                    if hasattr(trace, 'orientation'):
                        trace_info["orientation"] = trace.orientation
                
                traces.append(trace_info)
            
            figure_data["traces"] = traces
        
        # Extract layout information
        if hasattr(fig, 'layout'):
            layout = {}
            if hasattr(fig.layout, 'title') and hasattr(fig.layout.title, 'text'):
                layout["title"] = fig.layout.title.text
            if hasattr(fig.layout, 'xaxis') and hasattr(fig.layout.xaxis, 'title') and hasattr(fig.layout.xaxis.title, 'text'):
                layout["x_axis_title"] = fig.layout.xaxis.title.text
            if hasattr(fig.layout, 'yaxis') and hasattr(fig.layout.yaxis, 'title') and hasattr(fig.layout.yaxis.title, 'text'):
                layout["y_axis_title"] = fig.layout.yaxis.title.text
            
            figure_data["layout"] = layout
        
        # Handle specific chart types
        if "violin" in str(figure_data).lower():
            figure_data["visualization_details"] = """
            This is a violin plot which shows:
            1. Distribution shape using kernel density estimation (the outer "violin" shape)
            2. Box plot inside showing median, quartiles, and range
            3. The width at each point represents the density of data at that value
            4. Wider sections indicate more data points at that value
            5. The kernel density estimation smooths the data to show its distribution
            """
            
        elif "box" in str(figure_data).lower():
            figure_data["visualization_details"] = """
            This is a box plot showing:
            1. Median (center line)
            2. First and third quartiles (box boundaries)
            3. Whiskers typically extend to the most extreme data points within 1.5 times the interquartile range
            4. Outliers shown as individual points beyond the whiskers
            """
        
        # Convert to JSON-safe format using custom encoder
        return figure_data
        
    except Exception as e:
        return f"Error extracting figure data: {str(e)}"

def analyze_violin_plot_data(data_df, viz_params):
    """Extract KDE and distribution information from violin plot data"""
    if not viz_params or not data_df is not None:
        return "No data available for analysis"
    
    try:
        params = viz_params.get("parameters", {})
        if "x_column" in params and "y_column" in params:
            x_col = params["x_column"]
            y_col = params["y_column"]
            
            # Generate statistics by group
            group_stats = data_df.groupby(x_col)[y_col].agg(['count', 'mean', 'median', 'std', 'min', 'max']).reset_index()
            
            # Calculate IQR and range for each group
            results = []
            kde_analysis = []
            
            for group in data_df[x_col].unique():
                group_data = data_df[data_df[x_col] == group][y_col]
                
                # Calculate quartiles
                q1 = group_data.quantile(0.25)
                q3 = group_data.quantile(0.75)
                iqr = q3 - q1
                
                # Calculate range
                data_range = group_data.max() - group_data.min()
                
                # Calculate density metrics
                count = len(group_data)
                
                # Simple proxy for KDE peak: how concentrated the data is around the median
                median = group_data.median()
                mean = group_data.mean()
                std = group_data.std()
                
                # Values close to median indicate higher concentration
                concentration = count / (std if std > 0 else 1)  # Higher value = more concentrated
                
                # Calculate width of middle 50% of data (IQR)
                relative_concentration = iqr / data_range if data_range > 0 else 0  # Lower value = more concentrated
                
                # Estimate kde peak height (inversely related to std deviation)
                kde_peak_estimate = 1 / (std if std > 0 else 1)  # Higher value = sharper peak
                
                kde_analysis.append({
                    "group": group,
                    "count": count,
                    "mean": mean,
                    "median": median,
                    "std": std,
                    "iqr": iqr,
                    "range": data_range,
                    "concentration": concentration,
                    "relative_concentration": relative_concentration,
                    "kde_peak_estimate": kde_peak_estimate
                })
            
            # Find the group with highest concentration (proxy for highest KDE)
            if kde_analysis:
                highest_kde_group = max(kde_analysis, key=lambda x: x["kde_peak_estimate"])
                lowest_kde_group = min(kde_analysis, key=lambda x: x["kde_peak_estimate"])
                
                kde_insights = f"""
                KDE Analysis (Kernel Density Estimation):
                - Group with highest estimated KDE peak (most concentrated): {highest_kde_group['group']}
                - Group with lowest estimated KDE peak (most spread out): {lowest_kde_group['group']}
                
                KDE metrics per group:
                """
                
                for group_data in kde_analysis:
                    kde_insights += f"- {group_data['group']}: estimated peak height = {group_data['kde_peak_estimate']:.2f}, data concentration = {group_data['concentration']:.2f}\n"
                
                return kde_insights
            
            return "KDE analysis completed but no notable patterns found."
            
        return "Missing required columns for violin plot analysis"
    
    except Exception as e:
        return f"Error analyzing violin plot data: {str(e)}"

def generate_viz_response(question, api_key, visualization_data=None, insights=None, data_df=None, fig=None):
    """Generate a response about visualization and insights"""
    try:
        # Create a context with visualization information
        viz_context = ""
        if visualization_data:
            viz_context += f"Visualization Type: {visualization_data.get('viz_type', 'unknown')}\n"
            viz_context += "Visualization Parameters:\n"
            for key, value in visualization_data.get('parameters', {}).items():
                viz_context += f"- {key}: {value}\n"
        
        # Extract detailed figure data if available
        figure_details = ""
        if fig:
            figure_data = extract_figure_data(fig)
            # Convert to string safely, avoiding JSON serialization issues
            if isinstance(figure_data, dict):
                try:
                    figure_str = json.dumps(figure_data, indent=2, cls=NumpyEncoder)
                    figure_details = f"\nDetailed Figure Information:\n{figure_str}\n"
                except:
                    # Fallback to simpler representation if JSON serialization fails
                    figure_details = f"\nDetailed Figure Information: {str(figure_data)}\n"
            else:
                figure_details = f"\nDetailed Figure Information: {str(figure_data)}\n"
        
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
            
            # Add aggregated data based on visualization type
            if visualization_data and "viz_type" in visualization_data:
                viz_type = visualization_data["viz_type"]
                params = visualization_data.get("parameters", {})
                
                if viz_type == "violin_plot" and "x_column" in params and "y_column" in params:
                    x_col = params["x_column"]
                    y_col = params["y_column"]
                    
                    # Add summary stats by group
                    data_context += f"\nSummary Statistics of {y_col} by {x_col}:\n"
                    grouped_stats = data_df.groupby(x_col)[y_col].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
                    data_context += grouped_stats.to_string(index=False) + "\n"
                    
                    # Add KDE analysis
                    kde_analysis = analyze_violin_plot_data(data_df, visualization_data)
                    data_context += f"\n{kde_analysis}\n"
                    
                    # Add kernel density info
                    data_context += f"\nThe violin plot shows the kernel density estimation (KDE) for each {x_col} category. "
                    data_context += f"Wider parts of the violin indicate higher density of data points at those {y_col} values. "
                    data_context += f"The box plot inside shows the median, quartiles, and range of {y_col} values for each {x_col}."
        
        # Combine all context information
        full_context = viz_context + figure_details + insights_context + data_context
        
        # Extra context for kernel density in violin plots
        if "violin" in full_context.lower():
            full_context += """
            
            ABOUT KERNEL DENSITY ESTIMATION (KDE) IN VIOLIN PLOTS:
            Kernel Density Estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable. In violin plots:
            - The width of the violin at any point represents the estimated density of data at that value
            - Wider sections indicate more data points at that value
            - The KDE is essentially a smoothed histogram
            - The shape shows how the data is distributed
            - Categories with higher peaks in their KDE have more concentrated data at those values
            - Categories with wider KDEs have more spread-out distributions
            
            Understanding KDE peak height:
            - A high, narrow peak in the KDE indicates many data points concentrated at a specific value
            - The group with the highest KDE peak has the most concentrated distribution
            - Groups with lower, wider KDEs have more evenly spread distributions
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
If you're discussing a violin plot, be sure to explain aspects like the kernel density estimation (KDE), which is the outer shape showing data distribution.
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
