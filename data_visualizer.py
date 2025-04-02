import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import streamlit as st


class DataVisualizer:
    def __init__(self, df):
        """
        Initialize the DataVisualizer with a pandas DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data to visualize
        """
        self.df = df
        self.numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_columns = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col])]

    def get_column_types(self):
        """Return dictionary of column types for UI selection."""
        return {
            "Numeric": self.numeric_columns,
            "Categorical": self.categorical_columns,
            "Date": self.date_columns
        }

    def create_visualization(self, viz_type, **kwargs):
        """
        Create a visualization based on the specified type and parameters.

        Args:
            viz_type (str): Type of visualization to create
            **kwargs: Additional parameters specific to the visualization type

        Returns:
            plotly.graph_objects.Figure: The visualization figure
        """
        viz_functions = {
            "bar_chart": self._create_bar_chart,
            "line_chart": self._create_line_chart,
            "scatter_plot": self._create_scatter_plot,
            "pie_chart": self._create_pie_chart,
            "histogram": self._create_histogram,
            "box_plot": self._create_box_plot,
            "heatmap": self._create_heatmap,
            "time_series": self._create_time_series,
            "grouped_bar": self._create_grouped_bar,
            "stacked_bar": self._create_stacked_bar,
            "area_chart": self._create_area_chart,
            "bubble_chart": self._create_bubble_chart,
            "violin_plot": self._create_violin_plot,
            "sankey_diagram": self._create_sankey_diagram
        }

        if viz_type not in viz_functions:
            raise ValueError(f"Visualization type '{viz_type}' not supported")

        return viz_functions[viz_type](**kwargs)

    def _create_bar_chart(self, x_column, y_column, title=None, color_column=None, orientation='v'):
        """Create a bar chart."""
        if orientation == 'v':
            fig = px.bar(
                self.df,
                x=x_column,
                y=y_column,
                color=color_column,
                title=title or f"Bar Chart of {y_column} by {x_column}",
                labels={x_column: x_column.replace('_', ' ').title(),
                        y_column: y_column.replace('_', ' ').title()}
            )
        else:  # horizontal
            fig = px.bar(
                self.df,
                x=y_column,
                y=x_column,
                color=color_column,
                title=title or f"Bar Chart of {y_column} by {x_column}",
                labels={x_column: x_column.replace('_', ' ').title(),
                        y_column: y_column.replace('_', ' ').title()},
                orientation='h'
            )

        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title() if orientation == 'v' else y_column.replace('_',
                                                                                                       ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title() if orientation == 'v' else x_column.replace('_',
                                                                                                       ' ').title(),
            legend_title_text=color_column.replace('_', ' ').title() if color_column else None
        )

        return fig

    def _create_line_chart(self, x_column, y_column, title=None, color_column=None):
        """Create a line chart."""
        fig = px.line(
            self.df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=title or f"Line Chart of {y_column} over {x_column}",
            labels={x_column: x_column.replace('_', ' ').title(),
                    y_column: y_column.replace('_', ' ').title()}
        )

        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title(),
            legend_title_text=color_column.replace('_', ' ').title() if color_column else None
        )

        return fig

    def _create_scatter_plot(self, x_column, y_column, title=None, color_column=None, size_column=None):
        """Create a scatter plot."""
        fig = px.scatter(
            self.df,
            x=x_column,
            y=y_column,
            color=color_column,
            size=size_column,
            title=title or f"Scatter Plot of {y_column} vs {x_column}",
            labels={x_column: x_column.replace('_', ' ').title(),
                    y_column: y_column.replace('_', ' ').title()}
        )

        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title(),
            legend_title_text=color_column.replace('_', ' ').title() if color_column else None
        )

        return fig

    def _create_pie_chart(self, names_column, values_column, title=None):
        """Create a pie chart."""
        # Group data if needed
        if self.df[names_column].nunique() > 15:  # Too many categories for a pie chart
            st.warning(f"Too many categories ({self.df[names_column].nunique()}) for a pie chart. Showing top 15.")
            grouped_data = self.df.groupby(names_column)[values_column].sum().nlargest(15).reset_index()
        else:
            grouped_data = self.df.groupby(names_column)[values_column].sum().reset_index()

        fig = px.pie(
            grouped_data,
            names=names_column,
            values=values_column,
            title=title or f"Distribution of {values_column} by {names_column}",
            labels={names_column: names_column.replace('_', ' ').title(),
                    values_column: values_column.replace('_', ' ').title()}
        )

        return fig

    def _create_histogram(self, column, title=None, bins=None, color_column=None):
        """Create a histogram."""
        fig = px.histogram(
            self.df,
            x=column,
            color=color_column,
            nbins=bins,
            title=title or f"Histogram of {column}",
            labels={column: column.replace('_', ' ').title()}
        )

        fig.update_layout(
            xaxis_title=column.replace('_', ' ').title(),
            yaxis_title="Count",
            legend_title_text=color_column.replace('_', ' ').title() if color_column else None
        )

        return fig

    def _create_box_plot(self, x_column, y_column, title=None, color_column=None):
        """Create a box plot."""
        fig = px.box(
            self.df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=title or f"Box Plot of {y_column} by {x_column}",
            labels={x_column: x_column.replace('_', ' ').title(),
                    y_column: y_column.replace('_', ' ').title()}
        )

        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title(),
            legend_title_text=color_column.replace('_', ' ').title() if color_column else None
        )

        return fig

    def _create_heatmap(self, x_column, y_column, z_column, title=None):
        """Create a heatmap."""
        # Pivot data
        heatmap_data = self.df.pivot_table(index=y_column, columns=x_column, values=z_column, aggfunc='mean')

        fig = px.imshow(
            heatmap_data,
            labels=dict(x=x_column.replace('_', ' ').title(),
                        y=y_column.replace('_', ' ').title(),
                        color=z_column.replace('_', ' ').title()),
            title=title or f"Heatmap of {z_column} by {x_column} and {y_column}"
        )

        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title()
        )

        return fig

    def _create_time_series(self, date_column, value_column, title=None, color_column=None, resolution='day'):
        """Create a time series plot."""
        # Ensure date column is datetime
        df_copy = self.df.copy()
        if date_column not in self.date_columns:
            try:
                df_copy[date_column] = pd.to_datetime(df_copy[date_column])
            except:
                raise ValueError(f"Cannot convert {date_column} to datetime format")

        # Resample if needed
        if resolution != 'original':
            if color_column:
                grouped_dfs = []
                for name, group in df_copy.groupby(color_column):
                    resampled = self._resample_time_series(group, date_column, value_column, resolution)
                    resampled[color_column] = name
                    grouped_dfs.append(resampled)
                df_plot = pd.concat(grouped_dfs)
            else:
                df_plot = self._resample_time_series(df_copy, date_column, value_column, resolution)
        else:
            df_plot = df_copy

        fig = px.line(
            df_plot,
            x=date_column,
            y=value_column,
            color=color_column,
            title=title or f"Time Series of {value_column} over Time",
            labels={date_column: date_column.replace('_', ' ').title(),
                    value_column: value_column.replace('_', ' ').title()}
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=value_column.replace('_', ' ').title(),
            legend_title_text=color_column.replace('_', ' ').title() if color_column else None
        )

        return fig

    def _resample_time_series(self, df, date_column, value_column, resolution):
        """Helper function to resample time series data."""
        resample_map = {
            'day': 'D',
            'week': 'W',
            'month': 'M',
            'quarter': 'Q',
            'year': 'Y'
        }

        df = df.set_index(date_column)
        resampled = df[value_column].resample(resample_map[resolution]).mean().reset_index()
        return resampled

    def _create_grouped_bar(self, x_column, y_column, group_column, title=None):
        """Create a grouped bar chart."""
        fig = px.bar(
            self.df,
            x=x_column,
            y=y_column,
            color=group_column,
            barmode='group',
            title=title or f"Grouped Bar Chart of {y_column} by {x_column} and {group_column}",
            labels={x_column: x_column.replace('_', ' ').title(),
                    y_column: y_column.replace('_', ' ').title(),
                    group_column: group_column.replace('_', ' ').title()}
        )

        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title(),
            legend_title_text=group_column.replace('_', ' ').title()
        )

        return fig

    def _create_stacked_bar(self, x_column, y_column, stack_column, title=None):
        """Create a stacked bar chart."""
        fig = px.bar(
            self.df,
            x=x_column,
            y=y_column,
            color=stack_column,
            barmode='stack',
            title=title or f"Stacked Bar Chart of {y_column} by {x_column} and {stack_column}",
            labels={x_column: x_column.replace('_', ' ').title(),
                    y_column: y_column.replace('_', ' ').title(),
                    stack_column: stack_column.replace('_', ' ').title()}
        )

        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title(),
            legend_title_text=stack_column.replace('_', ' ').title()
        )

        return fig

    def _create_area_chart(self, x_column, y_column, title=None, group_column=None):
        """Create an area chart."""
        fig = px.area(
            self.df,
            x=x_column,
            y=y_column,
            color=group_column,
            title=title or f"Area Chart of {y_column} over {x_column}",
            labels={x_column: x_column.replace('_', ' ').title(),
                    y_column: y_column.replace('_', ' ').title()}
        )

        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title(),
            legend_title_text=group_column.replace('_', ' ').title() if group_column else None
        )

        return fig

    def _create_bubble_chart(self, x_column, y_column, size_column, title=None, color_column=None):
        """Create a bubble chart."""
        fig = px.scatter(
            self.df,
            x=x_column,
            y=y_column,
            size=size_column,
            color=color_column,
            title=title or f"Bubble Chart of {y_column} vs {x_column} sized by {size_column}",
            labels={x_column: x_column.replace('_', ' ').title(),
                    y_column: y_column.replace('_', ' ').title(),
                    size_column: size_column.replace('_', ' ').title()}
        )

        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title(),
            legend_title_text=color_column.replace('_', ' ').title() if color_column else None
        )

        return fig

    def _create_violin_plot(self, x_column, y_column, title=None, color_column=None):
        """Create a violin plot."""
        fig = px.violin(
            self.df,
            x=x_column,
            y=y_column,
            color=color_column,
            box=True,
            title=title or f"Violin Plot of {y_column} by {x_column}",
            labels={x_column: x_column.replace('_', ' ').title(),
                    y_column: y_column.replace('_', ' ').title()}
        )

        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title(),
            legend_title_text=color_column.replace('_', ' ').title() if color_column else None
        )

        return fig

    def _create_sankey_diagram(self, source_column, target_column, value_column, title=None):
        """Create a Sankey diagram."""
        # Group data to get the flow values
        df_flow = self.df.groupby([source_column, target_column])[value_column].sum().reset_index()

        # Create nodes
        all_nodes = pd.unique(df_flow[[source_column, target_column]].values.ravel('K'))
        node_indices = {node: i for i, node in enumerate(all_nodes)}

        # Create links
        source_indices = [node_indices[source] for source in df_flow[source_column]]
        target_indices = [node_indices[target] for target in df_flow[target_column]]
        values = df_flow[value_column].tolist()

        # Create figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values
            )
        )])

        fig.update_layout(
            title=title or f"Sankey Diagram of {value_column} from {source_column} to {target_column}",
            font=dict(size=12)
        )

        return fig

    def suggest_visualization(self):
        """Suggest appropriate visualizations based on data types."""
        suggestions = []

        # Check column types
        has_numeric = len(self.numeric_columns) > 0
        has_categorical = len(self.categorical_columns) > 0
        has_date = len(self.date_columns) > 0

        # If we have numeric and categorical columns
        if has_numeric and has_categorical:
            numeric_col = self.numeric_columns[0]
            categorical_col = self.categorical_columns[0]

            # Bar chart
            suggestions.append({
                "title": "Bar Chart of Numeric by Category",
                "viz_type": "bar_chart",
                "params": {
                    "x_column": categorical_col,
                    "y_column": numeric_col,
                    "title": f"Bar Chart of {numeric_col} by {categorical_col}"
                }
            })

            # Box plot
            suggestions.append({
                "title": "Box Plot to See Distribution",
                "viz_type": "box_plot",
                "params": {
                    "x_column": categorical_col,
                    "y_column": numeric_col,
                    "title": f"Box Plot of {numeric_col} by {categorical_col}"
                }
            })

        # If we have multiple numeric columns
        if len(self.numeric_columns) >= 2:
            # Scatter plot
            suggestions.append({
                "title": "Scatter Plot to See Relationships",
                "viz_type": "scatter_plot",
                "params": {
                    "x_column": self.numeric_columns[0],
                    "y_column": self.numeric_columns[1],
                    "title": f"Scatter Plot of {self.numeric_columns[1]} vs {self.numeric_columns[0]}"
                }
            })

            # Histogram
            suggestions.append({
                "title": "Histogram to See Distribution",
                "viz_type": "histogram",
                "params": {
                    "column": self.numeric_columns[0],
                    "title": f"Histogram of {self.numeric_columns[0]}"
                }
            })

        # If we have date and numeric columns
        if has_date and has_numeric:
            # Time series
            suggestions.append({
                "title": "Time Series Analysis",
                "viz_type": "time_series",
                "params": {
                    "date_column": self.date_columns[0],
                    "value_column": self.numeric_columns[0],
                    "title": f"Time Series of {self.numeric_columns[0]} over Time"
                }
            })

        # If we have multiple categorical columns and a numeric column
        if has_categorical and len(self.categorical_columns) >= 2 and has_numeric:
            # Grouped bar
            suggestions.append({
                "title": "Grouped Bar Chart for Comparison",
                "viz_type": "grouped_bar",
                "params": {
                    "x_column": self.categorical_columns[0],
                    "y_column": self.numeric_columns[0],
                    "group_column": self.categorical_columns[1],
                    "title": f"Grouped Bar Chart of {self.numeric_columns[0]} by {self.categorical_columns[0]} and {self.categorical_columns[1]}"
                }
            })

            # Heatmap if appropriate
            if self.df[self.categorical_columns[0]].nunique() <= 20 and self.df[
                self.categorical_columns[1]].nunique() <= 20:
                suggestions.append({
                    "title": "Heatmap to Visualize Patterns",
                    "viz_type": "heatmap",
                    "params": {
                        "x_column": self.categorical_columns[0],
                        "y_column": self.categorical_columns[1],
                        "z_column": self.numeric_columns[0],
                        "title": f"Heatmap of {self.numeric_columns[0]} by {self.categorical_columns[0]} and {self.categorical_columns[1]}"
                    }
                })

        return suggestions