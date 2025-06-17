# streamlit_app.py
import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np # For random array in placeholders
import os

# Configuration
FLASK_API_URL = "http://localhost:5000" # Ensure this matches your Flask app's host and port

st.set_page_config(layout="wide", page_title="CausalBox Toolkit")

st.title("🔬 CausalBox: A Causal Inference Toolkit")
st.markdown("Uncover causal relationships, simulate interventions, and estimate treatment effects.")

# --- Session State Initialization ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'processed_columns' not in st.session_state:
    st.session_state.processed_columns = None
if 'causal_graph_adj' not in st.session_state:
    st.session_state.causal_graph_adj = None
if 'causal_graph_nodes' not in st.session_state:
    st.session_state.causal_graph_nodes = None

# --- Data Preprocessing Module ---
st.header("1. Data Preprocessor 🧹")
st.write("Upload your CSV dataset or use a generated sample dataset.")

# Option to use generated sample dataset
if st.button("Use Sample Dataset (sample_dataset.csv)"):
    # In a real scenario, Streamlit would serve the file or you'd load it directly if local.
    # For this setup, we assume the Flask backend can access it or you manually upload it once.
    # For demonstration, we'll simulate loading a generic DataFrame.
    # In a full deployment, you'd have a mechanism to either:
    #   a) Have Flask serve the sample file, or
    #   b) Directly load it in Streamlit if the app and data are co-located.
    try:
        # Assuming the sample dataset is accessible or you are testing locally with `scripts/generate_data.py`
        # and then manually uploading this generated file.
        # For simplicity, we'll create a dummy df here if not actually uploaded.
        sample_df_path = "../data/sample_dataset.csv" # Path relative to main.py or Streamlit app execution
        if os.path.exists(sample_df_path):
             sample_df = pd.read_csv(sample_df_path)
             st.success(f"Loaded sample dataset from {sample_df_path}. Please upload this file if running from different directory.")
        else:
            st.warning("Sample dataset not found at ../data/sample_dataset.csv. Please run 'python scripts/generate_data.py' first or upload a file manually.")
            # Dummy DataFrame for demonstration if sample file isn't found
            sample_df = pd.DataFrame(np.random.rand(10, 5), columns=[f'col_{i}' for i in range(5)])

        # Convert to JSON for Flask API call
        files = {'file': ('sample_dataset.csv', sample_df.to_csv(index=False), 'text/csv')}
        response = requests.post(f"{FLASK_API_URL}/preprocess/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.processed_data = result['data']
            st.session_state.processed_columns = result['columns']
            st.success("Sample dataset preprocessed successfully!")
            st.dataframe(pd.DataFrame(st.session_state.processed_data).head()) # Display first few rows
        else:
            st.error(f"Error preprocessing sample dataset: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"Could not load or process sample dataset: {e}")


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    st.info("Uploading and preprocessing data...")
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
    try:
        response = requests.post(f"{FLASK_API_URL}/preprocess/upload", files=files)
        if response.status_code == 200:
            result = response.json()
            st.session_state.processed_data = result['data']
            st.session_state.processed_columns = result['columns']
            st.success("File preprocessed successfully!")
            st.dataframe(pd.DataFrame(st.session_state.processed_data).head()) # Display first few rows
        else:
            st.error(f"Error during preprocessing: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to Flask API at {FLASK_API_URL}. Please ensure the backend is running.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# --- Causal Discovery Module ---
st.header("2. Causal Discovery 🕵️‍♂️")
if st.session_state.processed_data:
    st.write("Learn the causal structure from your preprocessed data.")
    
    discovery_algo = st.selectbox(
        "Select Causal Discovery Algorithm:",
        ("PC Algorithm", "GES (Greedy Equivalence Search) - Placeholder", "NOTEARS - Placeholder")
    )
    
    if st.button("Discover Causal Graph"):
        st.info(f"Discovering graph using {discovery_algo}...")
        algo_map = {
            "PC Algorithm": "pc",
            "GES (Greedy Equivalence Search) - Placeholder": "ges",
            "NOTEARS - Placeholder": "notears"
        }
        selected_algo_code = algo_map[discovery_algo]

        try:
            response = requests.post(
                f"{FLASK_API_URL}/discover/",
                json={"data": st.session_state.processed_data, "algorithm": selected_algo_code}
            )
            if response.status_code == 200:
                result = response.json()
                st.session_state.causal_graph_adj = result['graph']
                st.session_state.causal_graph_nodes = st.session_state.processed_columns
                st.success("Causal graph discovered!")
                st.subheader("Causal Graph Visualization")
                # Visualization will be handled by the Causal Graph Visualizer section
            else:
                st.error(f"Error during causal discovery: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to Flask API at {FLASK_API_URL}. Please ensure the backend is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Please preprocess data first to enable causal discovery.")

# --- Causal Graph Visualizer Module ---
st.header("3. Causal Graph Visualizer 📊")
if st.session_state.causal_graph_adj and st.session_state.causal_graph_nodes:
    st.write("Interactive visualization of the discovered causal graph.")
    try:
        response = requests.post(
            f"{FLASK_API_URL}/visualize/graph",
            json={"graph": st.session_state.causal_graph_adj, "nodes": st.session_state.causal_graph_nodes}
        )
        if response.status_code == 200:
            graph_json = response.json()['graph']
            fig = go.Figure(json.loads(graph_json))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Graph Explanation:**
            * **Nodes:** Represent variables in your dataset.
            * **Arrows (Edges):** Indicate a direct causal influence from one variable (the tail) to another (the head).
            * **No Arrow:** Suggests no direct causal relationship was found, or the relationship is mediated by other variables.

            This graph helps answer "Why did it happen?" by showing the structural relationships.
            """)
        else:
            st.error(f"Error visualizing graph: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to Flask API at {FLASK_API_URL}. Please ensure the backend is running.")
    except Exception as e:
        st.error(f"An unexpected error occurred during visualization: {e}")
else:
    st.info("Please discover a causal graph first to visualize it.")


# --- Do-Calculus Engine Module ---
st.header("4. Do-Calculus Engine 🧪")
if st.session_state.processed_data and st.session_state.causal_graph_adj:
    st.write("Simulate interventions and observe their effects based on the causal graph.")
    
    intervention_var = st.selectbox(
        "Select variable to intervene on:",
        st.session_state.processed_columns,
        key="inter_var_select"
    )
    # Attempt to infer type for intervention_value input
    # Simplified approach: assuming numerical for now due to preprocessor output
    if intervention_var and isinstance(st.session_state.processed_data[0][intervention_var], (int, float)):
        intervention_value = st.number_input(f"Set '{intervention_var}' to value:", key="inter_val_input")
    else: # Treat as string/categorical for input, then try to preprocess for API
        intervention_value = st.text_input(f"Set '{intervention_var}' to value:", key="inter_val_input_text")
        st.warning("Categorical intervention values might require specific encoding logic on the backend.")

    if st.button("Perform Intervention"):
        st.info(f"Performing intervention: do('{intervention_var}' = {intervention_value})...")
        try:
            response = requests.post(
                f"{FLASK_API_URL}/intervene/",
                json={
                    "data": st.session_state.processed_data,
                    "intervention_var": intervention_var,
                    "intervention_value": intervention_value,
                    "graph": st.session_state.causal_graph_adj # Pass graph for advanced do-calculus
                }
            )
            if response.status_code == 200:
                intervened_data = pd.DataFrame(response.json()['intervened_data'])
                st.success("Intervention simulated successfully!")
                st.subheader("Intervened Data (First 10 rows)")
                st.dataframe(intervened_data.head(10))
                
                # Simple comparison visualization (e.g., histogram of outcome variable)
                if st.session_state.processed_columns and 'FinalExamScore' in st.session_state.processed_columns:
                    original_df = pd.DataFrame(st.session_state.processed_data)
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(x=original_df['FinalExamScore'], name='Original', opacity=0.7))
                    fig_dist.add_trace(go.Histogram(x=intervened_data['FinalExamScore'], name='Intervened', opacity=0.0))
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
                    st.markdown("""
                    **Intervention Explanation:**
                    * By simulating `do(X=x)`, we are forcing the value of X, effectively breaking its causal links from its parents.
                    * The graph above shows the distribution of a key outcome variable (e.g., `FinalExamScore`) before and after the intervention.
                    * This helps answer "What if we do this instead?" by showing the predicted outcome.
                    """)
                else:
                    st.info("Consider adding a relevant outcome variable to your dataset for better intervention analysis.")
            else:
                st.error(f"Error during intervention: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to Flask API at {FLASK_API_URL}. Please ensure the backend is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred during intervention: {e}")
else:
    st.info("Please preprocess data and discover a causal graph first to perform interventions.")

# --- Treatment Effect Estimator Module ---
st.header("5. Treatment Effect Estimator 🎯")
if st.session_state.processed_data:
    st.write("Estimate Average Treatment Effect (ATE) or Conditional Treatment Effect (CATE).")
    
    col1, col2 = st.columns(2)
    with col1:
        treatment_col = st.selectbox(
            "Select Treatment Variable:",
            st.session_state.processed_columns,
            key="treat_col_select"
        )
    with col2:
        outcome_col = st.selectbox(
            "Select Outcome Variable:",
            st.session_state.processed_columns,
            key="outcome_col_select"
        )
    
    all_cols_except_treat_outcome = [col for col in st.session_state.processed_columns if col not in [treatment_col, outcome_col]]
    covariates = st.multiselect(
        "Select Covariates (confounders):",
        all_cols_except_treat_outcome,
        default=all_cols_except_treat_outcome, # Default to all other columns
        key="covariates_select"
    )

    estimation_method = st.selectbox(
        "Select Estimation Method:",
        (
            "Linear Regression ATE",
            "Propensity Score Matching - Placeholder",
            "Inverse Propensity Weighting - Placeholder",
            "T-learner - Placeholder",
            "S-learner - Placeholder"
        )
    )

    if st.button("Estimate Treatment Effect"):
        st.info(f"Estimating treatment effect using {estimation_method}...")
        method_map = {
            "Linear Regression ATE": "linear_regression",
            "Propensity Score Matching - Placeholder": "propensity_score_matching",
            "Inverse Propensity Weighting - Placeholder": "inverse_propensity_weighting",
            "T-learner - Placeholder": "t_learner",
            "S-learner - Placeholder": "s_learner"
        }
        selected_method_code = method_map[estimation_method]

        try:
            response = requests.post(
                f"{FLASK_API_URL}/treatment/estimate_ate",
                json={
                    "data": st.session_state.processed_data,
                    "treatment_col": treatment_col,
                    "outcome_col": outcome_col,
                    "covariates": covariates,
                    "method": selected_method_code
                }
            )
            if response.status_code == 200:
                ate_result = response.json()['result']
                st.success(f"Treatment effect estimated using {estimation_method}:")
                st.write(f"**Estimated ATE: {ate_result:.4f}**")
                st.markdown("""
                **Treatment Effect Explanation:**
                * **Average Treatment Effect (ATE):** Measures the average causal effect of a treatment (e.g., `StudyHours`) on an outcome (e.g., `FinalExamScore`) across the entire population.
                * It answers "How much does doing X cause a change in Y?".
                * This estimation attempts to control for confounders (variables that influence both treatment and outcome) to isolate the true causal effect.
                """)
            else:
                st.error(f"Error during ATE estimation: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to Flask API at {FLASK_API_URL}. Please ensure the backend is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred during ATE estimation: {e}")
else:
    st.info("Please preprocess data first to estimate treatment effects.")

# --- Optional Advanced Add-Ons (Future Considerations) ---
st.header("Optional Advanced Add-Ons (Future Work) 🚀")
st.markdown("""
-   **🔄 Auto-causal graph refresh if dataset updates:** This would involve setting up a background process (e.g., using `watchfiles` with a separate service or integrated carefully into Flask/Streamlit) that monitors changes to the source CSV file. Upon detection, it would re-run the preprocessing and causal discovery, updating the dashboard live. This requires more complex architecture (e.g., WebSockets for real-time updates to Streamlit or scheduled background tasks).
-   **🕰️ Time-Series Causal Discovery (e.g., Granger Causality):** This requires handling time-indexed data and implementing algorithms specifically designed for temporal causal relationships. It would involve a separate data input and discovery module.
""")

st.markdown("---")
st.info("Developed by CausalBox Team. For support, please contact us.")