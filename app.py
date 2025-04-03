import streamlit as st
import pandas as pd
import os
import tempfile
import time
from pathlib import Path
import sys
import base64
from token_usage_tracker import TokenUsageTracker

# Import the DataAnalyzer class
from data_analyzer_fixed import DataAnalyzer

# Import the chat interface
from chat_interface import add_chat_to_ask_questions_tab

# Set page configuration
st.set_page_config(
    page_title="Excel Data Analyzer with OpenAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #444;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #0066cc;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #f0fff0;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #00cc66;
        margin-bottom: 20px;
    }
    .stButton button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
    }
    .file-info {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'excel_data' not in st.session_state:
    st.session_state.excel_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'temp_dirs' not in st.session_state:
    st.session_state.temp_dirs = []


def main():
    # Main header
    st.markdown('<div class="main-header">Excel Data Analyzer with OpenAI</div>', unsafe_allow_html=True)

    # Initialize token usage tracker
    token_tracker = TokenUsageTracker()

    # Display token usage in sidebar
    token_tracker.display_usage_sidebar()

    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### Configuration")

        # OpenAI API Key input
        api_key = st.text_input("OpenAI API Key", type="password",
                                help="Your OpenAI API key. It will not be stored permanently.")

        # OpenAI Model selection
        model = st.selectbox(
            "Select OpenAI Model",
            ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            index=0,
            help="Choose which OpenAI model to use for analysis."
        )

        # Additional options
        st.markdown("### Options")
        show_excel_preview = st.checkbox("Show Excel Preview", value=True)
        show_raw_context = st.checkbox("Show Raw Context (Advanced)", value=False)
        # Store in session state for use in chat interface
        st.session_state.show_raw_context = show_raw_context

        # About section
        st.markdown("### About")
        st.markdown("""
        This app uses OpenAI to analyze Excel data with context from SOPs and PDF documents.

        Upload your Excel file and supporting documents to get started.
        """)

        # Store token tracker in session state for use in other functions
        st.session_state.token_tracker = token_tracker

    # File upload section
    st.markdown('<div class="sub-header">1. Upload Your Files</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        excel_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"],
                                      help="The Excel file containing the data to analyze.")

    with col2:
        sop_files = st.file_uploader("Upload SOP Files", type=["txt", "pdf"],
                                     accept_multiple_files=True,
                                     help="Standard Operating Procedures for context.")

    with col3:
        pdf_files = st.file_uploader("Upload Reference PDFs", type=["pdf"],
                                     accept_multiple_files=True,
                                     help="Additional PDF documents for context.")

    # Save uploaded files to temporary directories
    temp_dir = None
    if excel_file is not None and (sop_files or pdf_files):
        # Set API key from input
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # Create temporary directories for the uploaded files
        temp_dir = tempfile.TemporaryDirectory()
        st.session_state.temp_dirs.append(temp_dir)

        # Create subdirectories
        temp_path = Path(temp_dir.name)
        sop_dir = temp_path / "sops"
        pdf_dir = temp_path / "pdfs"
        sop_dir.mkdir(exist_ok=True)
        pdf_dir.mkdir(exist_ok=True)

        # Save Excel file
        excel_path = temp_path / excel_file.name
        with open(excel_path, "wb") as f:
            f.write(excel_file.getvalue())

        # Save SOP files
        for sop in sop_files:
            sop_path = sop_dir / sop.name
            with open(sop_path, "wb") as f:
                f.write(sop.getvalue())

        # Save PDF files
        for pdf in pdf_files:
            pdf_path = pdf_dir / pdf.name
            with open(pdf_path, "wb") as f:
                f.write(pdf.getvalue())

        # Display file information
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Files Uploaded")
        st.markdown(f"📊 Excel File: **{excel_file.name}**")

        if sop_files:
            st.markdown(f"📄 SOP Files: **{len(sop_files)}** files")
            if st.expander("View SOP Files"):
                for sop in sop_files:
                    st.markdown(f"- {sop.name}")

        if pdf_files:
            st.markdown(f"📚 PDF References: **{len(pdf_files)}** files")
            if st.expander("View PDF Files"):
                for pdf in pdf_files:
                    st.markdown(f"- {pdf.name}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Initialize analyzer
        try:
            with st.spinner("Initializing data analyzer..."):
                # Create the analyzer
                st.session_state.analyzer = DataAnalyzer(
                    excel_path=str(excel_path),
                    sop_dir=str(sop_dir),
                    pdf_dir=str(pdf_dir),
                    model=model
                )

                # Load Excel data for preview
                st.session_state.excel_data = pd.read_excel(excel_path)

            st.markdown('<div class="success-box">✅ Analyzer initialized successfully!</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error initializing analyzer: {str(e)}")
            st.session_state.analyzer = None

    # Display Excel data preview with filtering options
    if st.session_state.excel_data is not None and show_excel_preview:
        st.markdown('<div class="sub-header">Excel Data Preview and Filtering</div>', unsafe_allow_html=True)

        # Initialize filtered_data in session state if not present
        if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = st.session_state.excel_data.copy()
            st.session_state.filters_applied = False

        # Add filtering section
        with st.expander("Filter Data", expanded=True):
            st.write("Apply filters to your Excel data. Analysis will use the filtered dataset.")

            # Column selector for filtering
            filter_cols = st.multiselect(
                "Select columns to filter by",
                options=st.session_state.excel_data.columns.tolist(),
                help="Choose which columns you want to filter on"
            )

            # Create filters for selected columns
            filters_changed = False
            temp_filtered_data = st.session_state.excel_data.copy()

            if filter_cols:
                for col in filter_cols:
                    col_type = st.session_state.excel_data[col].dtype

                    # Numeric column filtering
                    if pd.api.types.is_numeric_dtype(col_type):
                        min_val = float(st.session_state.excel_data[col].min())
                        max_val = float(st.session_state.excel_data[col].max())

                        # Avoid identical min/max values
                        if min_val == max_val:
                            min_val = 0.99 * min_val if min_val != 0 else -0.01
                            max_val = 1.01 * max_val if max_val != 0 else 0.01

                        # Create a slider for numeric filtering
                        filter_range = st.slider(
                            f"Filter by {col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            step=(max_val - min_val) / 100,
                            format="%.2f"
                        )

                        # Apply filter
                        temp_filtered_data = temp_filtered_data[(temp_filtered_data[col] >= filter_range[0]) &
                                                                (temp_filtered_data[col] <= filter_range[1])]

                    # Categorical column filtering
                    elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_categorical_dtype(col_type):
                        unique_values = st.session_state.excel_data[col].dropna().unique().tolist()
                        selected_values = st.multiselect(
                            f"Filter by {col}",
                            options=unique_values,
                            default=unique_values
                        )

                        if selected_values:
                            temp_filtered_data = temp_filtered_data[temp_filtered_data[col].isin(selected_values)]

                    # Date column filtering
                    elif pd.api.types.is_datetime64_dtype(col_type):
                        min_date = st.session_state.excel_data[col].min().date()
                        max_date = st.session_state.excel_data[col].max().date()

                        start_date = st.date_input(
                            f"Start date for {col}",
                            value=min_date
                        )
                        end_date = st.date_input(
                            f"End date for {col}",
                            value=max_date
                        )

                        temp_filtered_data = temp_filtered_data[
                            (temp_filtered_data[col].dt.date >= start_date) &
                            (temp_filtered_data[col].dt.date <= end_date)
                            ]

            # Apply the filters button
            if st.button("Apply Filters", key="apply_filters_button"):
                st.session_state.filtered_data = temp_filtered_data
                st.session_state.filters_applied = True
                filters_changed = True
                st.success(f"Filters applied! Dataset now contains {len(temp_filtered_data)} rows.")

            # Reset filters button
            if st.button("Reset Filters", key="reset_filters_button"):
                st.session_state.filtered_data = st.session_state.excel_data.copy()
                st.session_state.filters_applied = False
                filters_changed = True
                st.success("All filters reset. Using complete dataset.")

            # Show filter status
            if st.session_state.filters_applied:
                st.warning(
                    f"Filters are currently applied. Using {len(st.session_state.filtered_data)} of {len(st.session_state.excel_data)} rows.")

            # If filters changed, clear previous analysis results
            if filters_changed:
                st.session_state.analysis_results = None
                st.experimental_rerun()

        # Display the filtered data
        st.subheader("Data Preview")
        st.dataframe(st.session_state.filtered_data.head(100), use_container_width=True)

        # Show basic statistics of filtered data
        with st.expander("View Data Statistics"):
            st.write("### Data Statistics")

            # General info
            st.write(f"Total Rows: {len(st.session_state.excel_data)}")
            st.write(f"Filtered Rows: {len(st.session_state.filtered_data)}")
            st.write(f"Columns: {len(st.session_state.filtered_data.columns)}")

            # Numeric columns
            numeric_cols = st.session_state.filtered_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.write("### Numeric Columns Summary (Filtered Data)")
                st.dataframe(st.session_state.filtered_data[numeric_cols].describe(), use_container_width=True)

    # Analysis section (only show if analyzer is initialized)
    if st.session_state.analyzer is not None:
        st.markdown('<div class="sub-header">2. Analyze Your Data</div>', unsafe_allow_html=True)

        # Analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["General Analysis", "Ask Questions", "Generate Reports", "Visualizations", "NL Visualizations"])

        # Tab 1: General Analysis
        with tab1:
            st.write("Run a general analysis of your Excel data.")

            # Analysis focus (optional)
            analysis_focus = st.text_input("Analysis Focus (Optional)",
                                           placeholder="E.g., sales trends, inventory issues, etc.",
                                           help="Specify a particular aspect to focus the analysis on (leave blank for general analysis).")

            # Add option to use filtered data
            use_filtered_data = st.checkbox("Use filtered data for analysis",
                                            value=st.session_state.get('filters_applied', False),
                                            help="When checked, analysis will be performed only on the filtered dataset.")

            if st.button("Run Analysis", key="run_analysis_btn"):
                with st.spinner("Analyzing data..."):
                    try:
                        # Create a temporary modified analyzer for filtered data if needed
                        if use_filtered_data and 'filtered_data' in st.session_state:
                            # Save filtered data to a temporary file
                            temp_excel_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                            st.session_state.filtered_data.to_excel(temp_excel_file.name, index=False)
                            temp_excel_file.close()

                            # Create a new analyzer with the filtered data
                            temp_analyzer = DataAnalyzer(
                                excel_path=temp_excel_file.name,
                                sop_dir=st.session_state.analyzer.sop_dir,
                                pdf_dir=st.session_state.analyzer.pdf_dir,
                                model=st.session_state.analyzer.model
                            )

                            # Perform analysis with the filtered data
                            if analysis_focus:
                                analysis_result = temp_analyzer.analyze_data(analysis_focus)
                            else:
                                analysis_result = temp_analyzer.analyze_data()

                            # Clean up the temporary file
                            os.unlink(temp_excel_file.name)
                        else:
                            # Use the original analyzer
                            if analysis_focus:
                                analysis_result = st.session_state.analyzer.analyze_data(analysis_focus)
                            else:
                                analysis_result = st.session_state.analyzer.analyze_data()

                        st.session_state.analysis_results = analysis_result
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")

            # Display analysis results if available
            if st.session_state.analysis_results:
                st.markdown("### Analysis Results")
                st.markdown(st.session_state.analysis_results)

                # Download button for results
                if st.download_button(
                        label="Download Analysis",
                        data=st.session_state.analysis_results,
                        file_name="excel_analysis_results.txt",
                        mime="text/plain",
                        key="download_analysis_button"
                ):
                    st.success("Analysis downloaded!")

        # Tab 2: Ask Questions - UPDATED to use the chat interface
        with tab2:
            # Use the new chat interface instead of the old single-question interface
            add_chat_to_ask_questions_tab()

        # Tab 3: Generate Reports
        with tab3:
            st.write("Generate different types of reports based on your data.")

            # Report type selection
            report_type = st.selectbox(
                "Report Type",
                ["summary", "detailed", "executive"],
                format_func=lambda x: {
                    "summary": "Summary Report (Brief overview with key points)",
                    "detailed": "Detailed Report (In-depth analysis with methodology and recommendations)",
                    "executive": "Executive Report (Business-focused with actionable insights)"
                }.get(x)
            )

            # Add option to use filtered data
            use_filtered_data = st.checkbox("Use filtered data for report",
                                            value=st.session_state.get('filters_applied', False),
                                            help="When checked, report will be generated based only on the filtered dataset.")

            if st.button("Generate Report", key="gen_report_btn"):
                with st.spinner(f"Generating {report_type} report..."):
                    try:
                        # Handle filtered data if selected
                        if use_filtered_data and 'filtered_data' in st.session_state:
                            # Save filtered data to a temporary file
                            temp_excel_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                            st.session_state.filtered_data.to_excel(temp_excel_file.name, index=False)
                            temp_excel_file.close()

                            # Create a temporary analyzer with the filtered data
                            temp_analyzer = DataAnalyzer(
                                excel_path=temp_excel_file.name,
                                sop_dir=st.session_state.analyzer.sop_dir,
                                pdf_dir=st.session_state.analyzer.pdf_dir,
                                model=st.session_state.analyzer.model
                            )

                            # Generate report using the filtered data
                            report = temp_analyzer.generate_report(report_type)

                            # Clean up the temporary file
                            os.unlink(temp_excel_file.name)
                        else:
                            # Use the original analyzer
                            report = st.session_state.analyzer.generate_report(report_type)

                        st.markdown(f"### {report_type.title()} Report")
                        st.markdown(report)

                        # Add note about filtered data if applicable
                        if use_filtered_data and 'filtered_data' in st.session_state:
                            st.info(
                                f"This report is based on the filtered dataset ({len(st.session_state.filtered_data)} rows) rather than the complete dataset ({len(st.session_state.excel_data)} rows).")

                        # Download button for report
                        if st.download_button(
                                label="Download Report",
                                data=report,
                                file_name=f"{report_type}_report.txt",
                                mime="text/plain",
                                key="download_report_button"
                        ):
                            st.success("Report downloaded!")
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")

        # Tab 4: Visualizations
        from visualization_tab import add_visualization_tab
        add_visualization_tab({"General Analysis": tab1, "Ask Questions": tab2, "Generate Reports": tab3,
                               "Visualizations": tab4, "NL Visualizations": tab5},
                              st.session_state.analyzer,
                              st.session_state)

        # Tab 5: Natural Language Visualizations
        from natural_language_viz import add_nl_visualization_tab
        add_nl_visualization_tab({"General Analysis": tab1, "Ask Questions": tab2, "Generate Reports": tab3,
                                  "Visualizations": tab4, "NL Visualizations": tab5},
                                 st.session_state.analyzer,
                                 st.session_state)

    # Footer
    st.markdown("---")
    st.markdown("Excel Data Analyzer with OpenAI • Powered by Streamlit")


# Cleanup function for temporary directories
def cleanup_temp_dirs():
    for temp_dir in st.session_state.temp_dirs:
        try:
            temp_dir.cleanup()
        except:
            pass


# Register the cleanup function to be called when the app exits
import atexit

atexit.register(cleanup_temp_dirs)

if __name__ == "__main__":
    main()
