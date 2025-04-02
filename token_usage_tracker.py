import streamlit as st
import tiktoken
import json
import time
import pandas as pd
from datetime import datetime


class TokenUsageTracker:
    def __init__(self):
        """Initialize the token usage tracker."""
        # Initialize session state for token tracking if not exists
        if 'token_usage' not in st.session_state:
            st.session_state.token_usage = {
                'total_prompt_tokens': 0,
                'total_completion_tokens': 0,
                'total_tokens': 0,
                'api_calls': 0,
                'cost': 0.0,
                'start_time': time.time(),
                'history': []
            }

        # OpenAI model pricing (per 1K tokens as of April 2025)
        self.model_pricing = {
            'gpt-3.5-turbo': {'prompt': 0.0015, 'completion': 0.0020},
            'gpt-4-turbo': {'prompt': 0.01, 'completion': 0.03},
            'gpt-4': {'prompt': 0.03, 'completion': 0.06},
            'gpt-4-32k': {'prompt': 0.06, 'completion': 0.12},
            'gpt-4o': {'prompt': 0.005, 'completion': 0.015}
        }

    def estimate_tokens(self, text, model="gpt-3.5-turbo"):
        """Estimate the number of tokens in the text."""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimate (1 token ≈ 4 chars for English text)
            return len(text) // 4

    def track_api_call(self, prompt, response, model="gpt-3.5-turbo"):
        """
        Track token usage and cost for an API call.

        Args:
            prompt (str): The prompt sent to the API
            response (str or dict): The response from the API (could be text or full JSON response)
            model (str): The model used for the API call
        """
        # Extract token counts
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        # If response is a JSON dict with usage info (direct API response)
        if isinstance(response, dict) and 'usage' in response:
            usage = response['usage']
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
        else:
            # Estimate tokens if not provided
            prompt_tokens = self.estimate_tokens(prompt, model)
            completion_tokens = self.estimate_tokens(response if isinstance(response, str) else json.dumps(response),
                                                     model)
            total_tokens = prompt_tokens + completion_tokens

        # Calculate cost
        model_price = self.model_pricing.get(model, self.model_pricing[
            'gpt-3.5-turbo'])  # default to gpt-3.5-turbo pricing if model not found
        prompt_cost = (prompt_tokens / 1000) * model_price['prompt']
        completion_cost = (completion_tokens / 1000) * model_price['completion']
        total_cost = prompt_cost + completion_cost

        # Update session state
        st.session_state.token_usage['total_prompt_tokens'] += prompt_tokens
        st.session_state.token_usage['total_completion_tokens'] += completion_tokens
        st.session_state.token_usage['total_tokens'] += total_tokens
        st.session_state.token_usage['api_calls'] += 1
        st.session_state.token_usage['cost'] += total_cost

        # Record in history
        st.session_state.token_usage['history'].append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': model,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'cost': total_cost
        })

        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'cost': total_cost
        }

    def display_usage_sidebar(self):
        """Display token usage and cost in the sidebar."""
        with st.sidebar.expander("💰 Token Usage & Cost", expanded=False):
            # Add refresh button at the top
            if st.button("🔄 Refresh Stats", key="refresh_token_stats"):
                # No action needed - the rerun happens automatically when button is pressed
                pass

            usage = st.session_state.token_usage

            # Show last updated time
            st.markdown(f"*Last updated: {datetime.now().strftime('%H:%M:%S')}*")

            st.markdown("### Session Summary")

            # Calculate session time
            session_duration = int(time.time() - usage['start_time'])
            hours, remainder = divmod(session_duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            session_time_str = f"{hours}h {minutes}m {seconds}s"

            # Create metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total API Calls", usage['api_calls'])
                st.metric("Total Tokens", f"{usage['total_tokens']:,}")

            with col2:
                st.metric("Session Duration", session_time_str)
                st.metric("Estimated Cost", f"${usage['cost']:.4f}")

            st.markdown("### Token Breakdown")
            st.markdown(f"**Prompt Tokens:** {usage['total_prompt_tokens']:,}")
            st.markdown(f"**Completion Tokens:** {usage['total_completion_tokens']:,}")

            # Show token usage history if there are entries
            if len(usage['history']) > 0:
                st.markdown("### Recent API Calls")
                history_df = pd.DataFrame(usage['history'])

                # Format the dataframe for display
                if len(history_df) > 10:
                    history_df = history_df.tail(10)  # Show only the last 10 calls

                # Format the cost column
                history_df['cost'] = history_df['cost'].apply(lambda x: f"${x:.4f}")

                # Rename columns for better readability
                history_df = history_df.rename(columns={
                    'timestamp': 'Time',
                    'model': 'Model',
                    'prompt_tokens': 'Prompt',
                    'completion_tokens': 'Completion',
                    'total_tokens': 'Total',
                    'cost': 'Cost'
                })

                st.dataframe(history_df)

            # Add a reset button
            if st.button("Reset Usage Stats", key="reset_usage_stats"):
                # Reset all values except start_time
                start_time = st.session_state.token_usage['start_time']
                st.session_state.token_usage = {
                    'total_prompt_tokens': 0,
                    'total_completion_tokens': 0,
                    'total_tokens': 0,
                    'api_calls': 0,
                    'cost': 0.0,
                    'start_time': start_time,
                    'history': []
                }
                st.success("Usage statistics have been reset.")

    def get_usage_summary(self):
        """Get a summary of token usage and cost."""
        usage = st.session_state.token_usage
        return {
            'api_calls': usage['api_calls'],
            'total_tokens': usage['total_tokens'],
            'prompt_tokens': usage['total_prompt_tokens'],
            'completion_tokens': usage['total_completion_tokens'],
            'cost': usage['cost'],
            'session_duration': int(time.time() - usage['start_time'])
        }