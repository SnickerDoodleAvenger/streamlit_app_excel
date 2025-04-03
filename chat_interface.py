import streamlit as st
import pandas as pd
import os

def add_chat_to_ask_questions_tab():
    """
    Setup and handle the chat interface for the Ask Questions tab
    """
    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat header
    st.write("Ask specific questions about your data and get AI-powered answers.")
    
    # Display the chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input for new question
    user_question = st.chat_input("Ask a question about your data...")
    
    # Add option to use filtered data
    use_filtered_data = st.checkbox("Use filtered data for questions",
                                    value=st.session_state.get('filters_applied', False),
                                    help="When checked, questions will be answered based only on the filtered dataset.")
    
    if user_question:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display user question in the current session
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Processing your question..."):
                try:
                    # Handle filtered data if selected
                    if use_filtered_data and 'filtered_data' in st.session_state:
                        # Use filtered data
                        import tempfile
                        import os
                        
                        # Save filtered data to a temporary file
                        temp_excel_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                        st.session_state.filtered_data.to_excel(temp_excel_file.name, index=False)
                        temp_excel_file.close()

                        # Create a temporary analyzer with the filtered data
                        temp_analyzer = st.session_state.analyzer.__class__(
                            excel_path=temp_excel_file.name,
                            sop_dir=st.session_state.analyzer.sop_dir,
                            pdf_dir=st.session_state.analyzer.pdf_dir,
                            model=st.session_state.analyzer.model
                        )

                        # Get answer using the filtered data
                        answer = temp_analyzer.answer_question(user_question)

                        # Show the context used if advanced option selected
                        if st.session_state.get('show_raw_context', False):
                            context = temp_analyzer.get_relevant_context(user_question)

                        # Clean up the temporary file
                        os.unlink(temp_excel_file.name)
                    else:
                        # Use the original analyzer
                        answer = st.session_state.analyzer.answer_question(user_question)

                        # Get context for display if needed
                        if st.session_state.get('show_raw_context', False):
                            context = st.session_state.analyzer.get_relevant_context(user_question)

                    st.markdown(answer)

                    # Add note about filtered data if applicable
                    if use_filtered_data and 'filtered_data' in st.session_state:
                        st.info(
                            f"This answer is based on the filtered dataset ({len(st.session_state.filtered_data)} rows) rather than the complete dataset ({len(st.session_state.excel_data)} rows).")

                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                    # Show the context used if advanced option selected
                    if st.session_state.get('show_raw_context', False):
                        with st.expander("View Context Used"):
                            st.text(context)
                except Exception as e:
                    error_message = f"Error processing question: {str(e)}"
                    st.error(error_message)
                    # Add error message to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": f"❌ {error_message}"})
    
    # Add a button to clear chat history
    if st.session_state.chat_history and st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()
