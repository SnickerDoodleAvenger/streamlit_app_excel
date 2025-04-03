import streamlit as st
import pandas as pd
import os
import tempfile

def add_chat_to_ask_questions_tab():
    """
    Setup and handle the chat interface for the Ask Questions tab
    """
    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat header
    st.write("Ask specific questions about your data and get AI-powered answers.")
    
    # Add option to use filtered data
    use_filtered_data = st.checkbox("Use filtered data for questions",
                                value=st.session_state.get('filters_applied', False),
                                help="When checked, questions will be answered based only on the filtered dataset.")
    
    # Add a button to clear chat history
    if st.session_state.chat_history and st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()
    
    # Display the chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Create a function to generate response and store the API context
    def generate_response(question):
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

                # Create a modified question that includes chat history context
                context_question = create_contextual_question(question)
                
                # Get answer using the filtered data
                answer = temp_analyzer.answer_question(context_question)

                # Get context for display if needed
                if st.session_state.get('show_raw_context', False):
                    raw_context = temp_analyzer.get_relevant_context(question)
                    # Store context for display if needed
                    st.session_state.last_raw_context = raw_context

                # Clean up the temporary file
                os.unlink(temp_excel_file.name)
            else:
                # Create a modified question that includes chat history context
                context_question = create_contextual_question(question)
                
                # Use the original analyzer
                answer = st.session_state.analyzer.answer_question(context_question)

                # Get context for display if needed
                if st.session_state.get('show_raw_context', False):
                    raw_context = st.session_state.analyzer.get_relevant_context(question)
                    # Store context for display if needed
                    st.session_state.last_raw_context = raw_context

            return answer
        except Exception as e:
            return f"Error processing question: {str(e)}"

    def create_contextual_question(new_question):
        """Create a question that includes context from previous conversation"""
        if len(st.session_state.chat_history) == 0:
            return new_question
            
        # Get the last few exchanges to provide context
        context_history = st.session_state.chat_history[-6:]  # Last 3 exchanges (3 questions, 3 answers)
        
        context_prompt = "The following is the conversation history between me and the user. Please be aware of this context when answering the latest question:\n\n"
        
        for msg in context_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_prompt += f"{role}: {msg['content']}\n\n"
            
        context_prompt += f"Now, please answer this latest question considering the above conversation history: {new_question}"
        
        return context_prompt
    
    # Input for new question
    user_question = st.chat_input("Ask a question about your data...")
    
    if user_question:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display user question in the current session
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Processing your question..."):
                answer = generate_response(user_question)
                st.markdown(answer)
                
                # Add note about filtered data if applicable
                if use_filtered_data and 'filtered_data' in st.session_state:
                    st.info(
                        f"This answer is based on the filtered dataset ({len(st.session_state.filtered_data)} rows) rather than the complete dataset ({len(st.session_state.excel_data)} rows).")
                
                # Show the context used if advanced option selected
                if st.session_state.get('show_raw_context', False) and hasattr(st.session_state, 'last_raw_context'):
                    with st.expander("View Context Used"):
                        st.text(st.session_state.last_raw_context)
                
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
