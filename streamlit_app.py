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
        sample_df_path = "data/sample_dataset.csv" # Path relative to main.py or Streamlit app execution
        if os.path.exists(sample_df_path):
             sample_df = pd.read_csv(sample_df_path)
             st.success(f"Loaded sample dataset from {sample_df_path}. Please upload this file if running from different directory.")
        else:
            st.warning("Sample dataset not found at data/sample_dataset.csv.")
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

# --- Prediction Module ---
st.header("6. Prediction Module 📈")
if st.session_state.processed_data:
    st.write("Train a machine learning model for prediction (Regression or Classification).")
    
    prediction_type = st.selectbox(
        "Select Prediction Type:",
        ("Regression", "Classification"),
        key="prediction_type_select"
    )

    all_columns = st.session_state.processed_columns
    
    suitable_target_columns = []
    if st.session_state.processed_data:
        temp_df = pd.DataFrame(st.session_state.processed_data)
        for col in all_columns:
            # For classification, check if column is object type (string), boolean,
            # or has a limited number of unique integer values (e.g., less than 20 unique values)
            if prediction_type == 'Classification':
                if temp_df[col].dtype == 'object' or temp_df[col].dtype == 'bool':
                    suitable_target_columns.append(col)
                elif pd.api.types.is_integer_dtype(temp_df[col]) and temp_df[col].nunique() < 20: # Heuristic for discrete integers
                     suitable_target_columns.append(col)
            # For regression, primarily numerical columns
            elif prediction_type == 'Regression':
                if pd.api.types.is_numeric_dtype(temp_df[col]):
                    suitable_target_columns.append(col)

    if not suitable_target_columns:
        st.warning(f"No suitable target columns found for {prediction_type}. Please check your data types.")
        target_col = None # Set to None to prevent error if no columns are found
    else:
        # Try to pre-select the currently chosen target_col if it's still suitable
        # Otherwise, default to the first suitable column
        if 'target_col_select' in st.session_state and st.session_state.target_col_select in suitable_target_columns:
            default_target_index = suitable_target_columns.index(st.session_state.target_col_select)
        else:
            default_target_index = 0

        target_col = st.selectbox(
            "Select Target Variable:",
            suitable_target_columns,
            index=default_target_index,
            key="target_col_select"
        )

    # Filter out the target column from feature options
    feature_options = [col for col in all_columns if col != target_col]
    feature_cols = st.multiselect(
        "Select Feature Variables:",
        feature_options,
        default=feature_options, # Default to all other columns
        key="feature_cols_select"
    )

    if st.button("Train Model & Predict", key="train_predict_button"):
        if not target_col or not feature_cols:
            st.warning("Please select a target variable and at least one feature variable.")
        else:
            st.info(f"Training {prediction_type} model using Random Forest...")
            try:
                response = requests.post(
                    f"{FLASK_API_URL}/prediction/train_predict",
                    json={
                        "data": st.session_state.processed_data,
                        "target_col": target_col,
                        "feature_cols": feature_cols,
                        "prediction_type": prediction_type.lower()
                    }
                )

                if response.status_code == 200:
                    results = response.json()['results']
                    st.success(f"{prediction_type} Model Trained Successfully!")
                    st.subheader("Model Performance")

                    if prediction_type == 'Regression':
                        st.write(f"**R-squared:** {results['r2_score']:.4f}")
                        st.write(f"**Mean Squared Error (MSE):** {results['mean_squared_error']:.4f}")
                        st.write(f"**Root Mean Squared Error (RMSE):** {results['root_mean_squared_error']:.4f}")

                        st.subheader("Actual vs. Predicted Plot")
                        actual_predicted_df = pd.DataFrame(results['actual_vs_predicted'])
                        fig_reg = px.scatter(actual_predicted_df, x='Actual', y='Predicted',
                                             title='Actual vs. Predicted Values',
                                             labels={'Actual': f'Actual {target_col}', 'Predicted': f'Predicted {target_col}'})
                        fig_reg.add_trace(go.Scatter(x=[actual_predicted_df['Actual'].min(), actual_predicted_df['Actual'].max()],
                                                     y=[actual_predicted_df['Actual'].min(), actual_predicted_df['Actual'].max()],
                                                     mode='lines', name='Ideal Fit', line=dict(dash='dash', color='red')))
                        st.plotly_chart(fig_reg, use_container_width=True)

                        st.subheader("Residual Plot")
                        actual_predicted_df['Residuals'] = actual_predicted_df['Actual'] - actual_predicted_df['Predicted']
                        fig_res = px.scatter(actual_predicted_df, x='Predicted', y='Residuals',
                                             title='Residual Plot',
                                             labels={'Predicted': f'Predicted {target_col}', 'Residuals': 'Residuals'})
                        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_res, use_container_width=True)

                    elif prediction_type == 'Classification':
                        st.write(f"**Accuracy:** {results['accuracy']:.4f}")
                        st.write(f"**Precision (weighted):** {results['precision']:.4f}")
                        st.write(f"**Recall (weighted):** {results['recall']:.4f}")
                        st.write(f"**F1-Score (weighted):** {results['f1_score']:.4f}")

                        st.subheader("Confusion Matrix")
                        conf_matrix = results['confusion_matrix']
                        class_labels = results.get('class_labels', [str(i) for i in range(len(conf_matrix))])
                        fig_cm = px.imshow(conf_matrix,
                                            labels=dict(x="Predicted", y="True", color="Count"),
                                            x=class_labels,
                                            y=class_labels,
                                            text_auto=True,
                                            color_continuous_scale="Viridis",
                                            title="Confusion Matrix")
                        st.plotly_chart(fig_cm, use_container_width=True)

                        st.subheader("Classification Report")
                        # Convert dict to DataFrame for nice display
                        report_df = pd.DataFrame(results['classification_report']).transpose()
                        st.dataframe(report_df)

                    st.subheader("Feature Importances")
                    feature_importances_df = pd.DataFrame(list(results['feature_importances'].items()), columns=['Feature', 'Importance'])
                    fig_fi = px.bar(feature_importances_df, x='Importance', y='Feature', orientation='h',
                                    title='Feature Importances',
                                    labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'})
                    fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort bars
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.error(f"Error during prediction: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to Flask API at {FLASK_API_URL}. Please ensure the backend is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
else:
    st.info("Please preprocess data first to use the Prediction Module.")

# --- Time Series Causal Discovery Module ---
st.header("7. Time Series Causal Discovery ⏰")
if st.session_state.processed_data:
    st.write("Infer causal relationships in time-series data using Granger Causality.")
    st.info("Ensure your dataset includes a timestamp column and that variables are numeric.")

    all_columns = st.session_state.processed_columns
    
    # Heuristic to suggest potential timestamp columns (object/string type, or first column)
    potential_ts_cols = [col for col in all_columns if pd.DataFrame(st.session_state.processed_data)[col].dtype == 'object']
    if not potential_ts_cols and all_columns: # If no object columns, suggest the first column
        potential_ts_cols = [all_columns[0]]
    
    timestamp_col = st.selectbox(
        "Select Timestamp Column:",
        potential_ts_cols if potential_ts_cols else ["No suitable timestamp column found. Please check data."],
        key="ts_col_select"
    )

    # Filter out timestamp column and non-numeric columns for analysis
    variables_for_ts_analysis = [
        col for col in all_columns if col != timestamp_col and pd.api.types.is_numeric_dtype(pd.DataFrame(st.session_state.processed_data)[col])
    ]

    variables_to_analyze = st.multiselect(
        "Select Variables to Analyze for Granger Causality:",
        variables_for_ts_analysis,
        default=variables_for_ts_analysis,
        key="ts_vars_select"
    )

    max_lags = st.number_input(
        "Max Lags (for Granger Causality):",
        min_value=1,
        value=5, # Default value
        step=1,
        help="The maximum number of lagged observations to consider for causality."
    )

    if st.button("Discover Time Series Causality", key="ts_discover_button"):
        if not timestamp_col or not variables_to_analyze:
            st.warning("Please select a timestamp column and at least one variable to analyze.")
        elif "No suitable timestamp column found" in timestamp_col:
            st.error("Cannot proceed. Please ensure your data has a suitable timestamp column.")
        else:
            st.info("Performing Granger Causality tests...")
            try:
                response = requests.post(
                    f"{FLASK_API_URL}/timeseries/discover_causality",
                    json={
                        "data": st.session_state.processed_data,
                        "timestamp_col": timestamp_col,
                        "variables_to_analyze": variables_to_analyze,
                        "max_lags": max_lags
                    }
                )

                if response.status_code == 200:
                    results = response.json()['results']
                    st.success("Time Series Causal Discovery Complete!")
                    st.subheader("Granger Causality Test Results")

                    if results:
                        # Convert results to a DataFrame for better display
                        results_df = pd.DataFrame(results)
                        results_df['p_value'] = results_df['p_value'].round(4) # Round p-values
                        st.dataframe(results_df)

                        st.markdown("**Interpretation:** A small p-value (typically < 0.05) suggests that the 'cause' variable Granger-causes the 'effect' variable. This means past values of the 'cause' variable help predict future values of the 'effect' variable, even when past values of the 'effect' variable are considered.")
                        st.markdown(f"*(Note: Granger Causality implies predictive causality, not necessarily true mechanistic causality. Also, ensure your time series are stationary for robust results.)*")

                        # Optionally, visualize a simple causality graph
                        st.subheader("Granger Causality Graph")
                        fig_ts_graph = go.Figure()
                        nodes = []
                        edges = []
                        edge_colors = []

                        # Add nodes
                        for i, var in enumerate(variables_to_analyze):
                            nodes.append(dict(id=var, label=var, x=np.cos(i*2*np.pi/len(variables_to_analyze)), y=np.sin(i*2*np.pi/len(variables_to_analyze))))

                        # Add edges
                        for res in results:
                            if res['p_value'] < 0.05: # Consider it a causal link if p-value is below significance
                                edges.append(dict(source=res['cause'], target=res['effect'], value=1/res['p_value'], title=f"p={res['p_value']:.4f}"))
                                edge_colors.append("blue")
                            else:
                                # Optional: Show non-significant edges in a different color or omit
                                pass
                        
                        # Use a simple network graph layout (Spring layout is common)
                        # For a truly interactive graph, you might need a different library or more complex Plotly setup
                        # This is a very basic attempt to visualize; consider more robust solutions like NetworkX + Plotly/Dash
                        
                        # Simple way to draw arrows for significant relationships
                        significant_edges = [edge for edge in results if edge['p_value'] < 0.05]
                        if significant_edges:
                            st.write("Visualizing significant (p < 0.05) Granger causal links:")
                            # This needs a more robust way to draw directed edges in plotly if using just scatter/lines.
                            # For now, let's just list them clearly.
                            for edge in significant_edges:
                                st.write(f"➡️ **{edge['cause']}** Granger-causes **{edge['effect']}** (p={edge['p_value']:.4f})")
                        else:
                            st.info("No significant Granger causal links found at p < 0.05.")

                    else:
                        st.info("No Granger Causality relationships found or data insufficient.")

                else:
                    st.error(f"Error during time-series causal discovery: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to Flask API at {FLASK_API_URL}. Please ensure the backend is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred during time-series causal discovery: {e}")
else:
    st.info("Please preprocess data first to use the Time Series Causal Discovery Module.")

# --- CausalBox Chat Assistant ---
st.header("8. CausalBox Chat Assistant 🤖")
st.write("Ask questions about your loaded dataset, causal concepts, or the discovered causal graph!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything about CausalBox..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare session context to send to the backend
    session_context = {
        "processed_data": st.session_state.processed_data,
        "processed_columns": st.session_state.processed_columns,
        "causal_graph_adj": st.session_state.causal_graph_adj,
        "causal_graph_nodes": st.session_state.causal_graph_nodes,
        # Add any other relevant session state variables that the chatbot might need
    }

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{FLASK_API_URL}/chatbot/message",
                json={
                    "user_message": prompt,
                    "session_context": session_context
                }
            )

            if response.status_code == 200:
                chatbot_response_text = response.json().get('response', 'Sorry, I could not generate a response.')
            else:
                chatbot_response_text = f"Error from chatbot backend: {response.json().get('detail', 'Unknown error')}"
        except requests.exceptions.ConnectionError:
            chatbot_response_text = f"Could not connect to Flask API at {FLASK_API_URL}. Please ensure the backend is running."
        except Exception as e:
            chatbot_response_text = f"An unexpected error occurred while getting chatbot response: {e}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(chatbot_response_text)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": chatbot_response_text})

# --- Future Work (Simplified) ---
st.header("Future Work 🚀")
st.markdown("""
-   **🔄 Auto-causal graph refresh:** Monitor dataset updates and automatically refresh the causal graph.
""")

st.markdown("---")
st.info("Developed by CausalBox Team. For support, please contact us.")