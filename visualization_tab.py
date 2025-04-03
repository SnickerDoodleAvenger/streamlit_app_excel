import streamlit as st
import os


def add_visualization_tab(app_tabs, analyzer, session_state):
    """Add a data visualization tab to the app."""
    from data_visualizer import DataVisualizer

    # Create a new tab for visualizations
    with app_tabs["Visualizations"]:
        st.write("Create visualizations based on your Excel data.")

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

        # Get column types for UI selection
        column_types = session_state.visualizer.get_column_types()

        # Create UI for visualization selection
        st.subheader("Choose Visualization Type")

        # Main visualization tabs
        viz_type_category = st.radio(
            "Visualization Category",
            ["Basic Charts", "Comparison Charts", "Distribution Charts", "Time Series", "Relationships",
             "Suggested Visualizations"],
            horizontal=True
        )

        # Different viz types based on the category
        if viz_type_category == "Basic Charts":
            viz_type = st.selectbox(
                "Chart Type",
                ["Bar Chart", "Line Chart", "Pie Chart", "Area Chart"]
            )
        elif viz_type_category == "Comparison Charts":
            viz_type = st.selectbox(
                "Chart Type",
                ["Grouped Bar Chart", "Stacked Bar Chart", "Box Plot", "Violin Plot"]
            )
        elif viz_type_category == "Distribution Charts":
            viz_type = st.selectbox(
                "Chart Type",
                ["Histogram", "Density Plot", "Box Plot"]
            )
        elif viz_type_category == "Time Series":
            viz_type = st.selectbox(
                "Chart Type",
                ["Time Series Line", "Time Series with Categories", "Cumulative Chart"]
            )
        elif viz_type_category == "Relationships":
            viz_type = st.selectbox(
                "Chart Type",
                ["Scatter Plot", "Bubble Chart", "Heatmap", "Sankey Diagram"]
            )
        else:  # Suggested Visualizations
            suggestions = session_state.visualizer.suggest_visualization()
            if not suggestions:
                st.warning("No visualization suggestions available for this dataset.")
                return

            # Display suggestions as selectable options
            suggestion_titles = [s["title"] for s in suggestions]
            selected_suggestion = st.selectbox(
                "Suggested Visualizations",
                suggestion_titles
            )

            # Find the selected suggestion
            selected_idx = suggestion_titles.index(selected_suggestion)
            suggestion = suggestions[selected_idx]

            # Create the visualization based on the suggestion
            st.subheader(suggestion["title"])
            fig = session_state.visualizer.create_visualization(
                suggestion["viz_type"],
                **suggestion["params"]
            )

            # Display the visualization
            st.plotly_chart(fig, use_container_width=True)

            # Download options
            st.download_button(
                label="Download Visualization as HTML",
                data=fig.to_html(),
                file_name=f"{suggestion['title'].replace(' ', '_')}.html",
                mime="text/html"
            )

            return  # Skip the rest of the function for suggestions

        # Create input UI based on visualization type
        st.subheader("Configure Visualization")

        params = {}

        if viz_type == "Bar Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-Axis (Categories)", column_types["Categorical"] or column_types["Numeric"])
                orientation = st.radio("Orientation", ["Vertical", "Horizontal"])
            with col2:
                y_column = st.selectbox("Y-Axis (Values)", column_types["Numeric"])
                color_column = st.selectbox("Color By (Optional)", ["None"] + column_types["Categorical"])

            params = {
                "viz_type": "bar_chart",
                "x_column": x_column,
                "y_column": y_column,
                "orientation": "v" if orientation == "Vertical" else "h",
                "title": st.text_input("Chart Title", value=f"Bar Chart of {y_column} by {x_column}")
            }
            if color_column != "None":
                params["color_column"] = color_column

        elif viz_type == "Line Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-Axis", column_types["Numeric"] + column_types["Date"])
                y_column = st.selectbox("Y-Axis", column_types["Numeric"])
            with col2:
                color_column = st.selectbox("Color By (Optional)", ["None"] + column_types["Categorical"])

            params = {
                "viz_type": "line_chart",
                "x_column": x_column,
                "y_column": y_column,
                "title": st.text_input("Chart Title", value=f"Line Chart of {y_column} over {x_column}")
            }
            if color_column != "None":
                params["color_column"] = color_column

        elif viz_type == "Pie Chart":
            col1, col2 = st.columns(2)
            with col1:
                names_column = st.selectbox("Categories", column_types["Categorical"])
            with col2:
                values_column = st.selectbox("Values", column_types["Numeric"])

            params = {
                "viz_type": "pie_chart",
                "names_column": names_column,
                "values_column": values_column,
                "title": st.text_input("Chart Title", value=f"Distribution of {values_column} by {names_column}")
            }

        elif viz_type == "Area Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-Axis", column_types["Numeric"] + column_types["Date"])
                y_column = st.selectbox("Y-Axis", column_types["Numeric"])
            with col2:
                group_column = st.selectbox("Group By (Optional)", ["None"] + column_types["Categorical"])

            params = {
                "viz_type": "area_chart",
                "x_column": x_column,
                "y_column": y_column,
                "title": st.text_input("Chart Title", value=f"Area Chart of {y_column} over {x_column}")
            }
            if group_column != "None":
                params["group_column"] = group_column

        elif viz_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-Axis", column_types["Numeric"])
                y_column = st.selectbox("Y-Axis", column_types["Numeric"])
            with col2:
                color_column = st.selectbox("Color By (Optional)", ["None"] + column_types["Categorical"])
                size_column = st.selectbox("Size By (Optional)", ["None"] + column_types["Numeric"])

            params = {
                "viz_type": "scatter_plot",
                "x_column": x_column,
                "y_column": y_column,
                "title": st.text_input("Chart Title", value=f"Scatter Plot of {y_column} vs {x_column}")
            }
            if color_column != "None":
                params["color_column"] = color_column
            if size_column != "None":
                params["size_column"] = size_column

        elif viz_type == "Bubble Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-Axis", column_types["Numeric"])
                y_column = st.selectbox("Y-Axis", column_types["Numeric"])
            with col2:
                size_column = st.selectbox("Bubble Size", column_types["Numeric"])
                color_column = st.selectbox("Color By (Optional)", ["None"] + column_types["Categorical"])

            params = {
                "viz_type": "bubble_chart",
                "x_column": x_column,
                "y_column": y_column,
                "size_column": size_column,
                "title": st.text_input("Chart Title",
                                       value=f"Bubble Chart of {y_column} vs {x_column} sized by {size_column}")
            }
            if color_column != "None":
                params["color_column"] = color_column

        elif viz_type == "Histogram":
            col1, col2 = st.columns(2)
            with col1:
                column = st.selectbox("Data Column", column_types["Numeric"])
                bins = st.slider("Number of Bins", min_value=5, max_value=100, value=20)
            with col2:
                color_column = st.selectbox("Group By (Optional)", ["None"] + column_types["Categorical"])

            params = {
                "viz_type": "histogram",
                "column": column,
                "bins": bins,
                "title": st.text_input("Chart Title", value=f"Histogram of {column}")
            }
            if color_column != "None":
                params["color_column"] = color_column

        elif viz_type == "Box Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Categories (X-Axis)", column_types["Categorical"])
                y_column = st.selectbox("Values (Y-Axis)", column_types["Numeric"])
            with col2:
                color_column = st.selectbox("Color By (Optional)", ["None"] + column_types["Categorical"])

            params = {
                "viz_type": "box_plot",
                "x_column": x_column,
                "y_column": y_column,
                "title": st.text_input("Chart Title", value=f"Box Plot of {y_column} by {x_column}")
            }
            if color_column != "None":
                params["color_column"] = color_column

        elif viz_type == "Violin Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Categories (X-Axis)", column_types["Categorical"])
                y_column = st.selectbox("Values (Y-Axis)", column_types["Numeric"])
            with col2:
                color_column = st.selectbox("Color By (Optional)", ["None"] + column_types["Categorical"])

            params = {
                "viz_type": "violin_plot",
                "x_column": x_column,
                "y_column": y_column,
                "title": st.text_input("Chart Title", value=f"Violin Plot of {y_column} by {x_column}")
            }
            if color_column != "None":
                params["color_column"] = color_column

        elif viz_type == "Heatmap":
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-Axis Categories", column_types["Categorical"])
                y_column = st.selectbox("Y-Axis Categories", column_types["Categorical"])
            with col2:
                z_column = st.selectbox("Values", column_types["Numeric"])

            params = {
                "viz_type": "heatmap",
                "x_column": x_column,
                "y_column": y_column,
                "z_column": z_column,
                "title": st.text_input("Chart Title", value=f"Heatmap of {z_column} by {x_column} and {y_column}")
            }

        elif viz_type == "Time Series Line" or viz_type == "Time Series with Categories":
            col1, col2 = st.columns(2)
            with col1:
                if column_types["Date"]:
                    date_column = st.selectbox("Date/Time Column", column_types["Date"])
                else:
                    date_column = st.selectbox("Date/Time Column (will be converted)",
                                               [col for col in data_for_viz.columns if
                                                "date" in col.lower() or "time" in col.lower() or "year" in col.lower() or "month" in col.lower()])

                value_column = st.selectbox("Value Column", column_types["Numeric"])

            with col2:
                resolution = st.selectbox("Time Resolution", ["original", "day", "week", "month", "quarter", "year"])
                if viz_type == "Time Series with Categories":
                    color_column = st.selectbox("Group By", column_types["Categorical"])
                else:
                    color_column = None

            params = {
                "viz_type": "time_series",
                "date_column": date_column,
                "value_column": value_column,
                "resolution": resolution,
                "title": st.text_input("Chart Title", value=f"Time Series of {value_column} over Time")
            }
            if color_column:
                params["color_column"] = color_column

        elif viz_type == "Grouped Bar Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-Axis Categories", column_types["Categorical"])
                y_column = st.selectbox("Y-Axis Values", column_types["Numeric"])
            with col2:
                group_column = st.selectbox("Group By", column_types["Categorical"])

            params = {
                "viz_type": "grouped_bar",
                "x_column": x_column,
                "y_column": y_column,
                "group_column": group_column,
                "title": st.text_input("Chart Title",
                                       value=f"Grouped Bar Chart of {y_column} by {x_column} and {group_column}")
            }

        elif viz_type == "Stacked Bar Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-Axis Categories", column_types["Categorical"])
                y_column = st.selectbox("Y-Axis Values", column_types["Numeric"])
            with col2:
                stack_column = st.selectbox("Stack By", column_types["Categorical"])

            params = {
                "viz_type": "stacked_bar",
                "x_column": x_column,
                "y_column": y_column,
                "stack_column": stack_column,
                "title": st.text_input("Chart Title",
                                       value=f"Stacked Bar Chart of {y_column} by {x_column} and {stack_column}")
            }

        elif viz_type == "Sankey Diagram":
            col1, col2 = st.columns(2)
            with col1:
                source_column = st.selectbox("Source Column", column_types["Categorical"])
                target_column = st.selectbox("Target Column", column_types["Categorical"])
            with col2:
                value_column = st.selectbox("Value Column", column_types["Numeric"])

            params = {
                "viz_type": "sankey_diagram",
                "source_column": source_column,
                "target_column": target_column,
                "value_column": value_column,
                "title": st.text_input("Chart Title",
                                       value=f"Sankey Diagram of {value_column} from {source_column} to {target_column}")
            }

       # Generate and display the visualization
        if st.button("Generate Visualization", key="standard_viz_generate_button"):
            try:
                with st.spinner("Creating visualization..."):
                    fig = session_state.visualizer.create_visualization(**params)
            
                    # Display the visualization
                    st.subheader("Visualization Result")
                    st.plotly_chart(fig, use_container_width=True)
            
                    # Download options
                    st.download_button(
                        label="Download Visualization as HTML",
                        data=fig.to_html(),
                        file_name=f"{params['viz_type']}_{params.get('title', 'visualization').replace(' ', '_')}.html",
                        mime="text/html",
                        key="download_viz_html"
                    )
            
                    # Get API key for insights
                    api_key = os.environ.get("OPENAI_API_KEY", "")
            
                    # Generate insights
                    with st.spinner("Generating insights..."):
                        insights = session_state.visualizer.generate_visualization_insights(
                            fig, 
                            params["viz_type"], 
                            params, 
                            session_state.filtered_data if session_state.filters_applied else session_state.excel_data,
                            api_key
                        )
                
                        # Display insights
                        st.subheader("Visualization Insights")
                        st.markdown(insights)
                
                        # Download insights
                        st.download_button(
                            label="Download Insights",
                            data=insights,
                            file_name=f"{params['viz_type']}_insights.md",
                            mime="text/markdown",
                            key="download_viz_insights"
                        )
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
