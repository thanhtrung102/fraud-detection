"""
Fraud Detection Streamlit UI
============================

Interactive web interface for fraud detection predictions.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.utils.model_loader import FraudModelLoader
from ui.utils.predictor import FraudPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🔍",
    layout="wide",
)

# Initialize session state
if "model_loader" not in st.session_state:
    st.session_state.model_loader = FraudModelLoader()
    st.session_state.predictor = FraudPredictor(st.session_state.model_loader)
    st.session_state.models_loaded = False
    st.session_state.run_id = None


# Header
st.title("🔍 Fraud Detection System")
st.markdown("Real-time credit card fraud detection using stacking ensemble models")

# Sidebar for model configuration
with st.sidebar:
    st.header("Model Configuration")

    # Model status
    if not st.session_state.models_loaded:
        st.warning("No model loaded")
    else:
        st.success("Model loaded")
        if st.session_state.run_id:
            st.caption(f"Run ID: {st.session_state.run_id[:12]}...")

        # Show model info
        summary = st.session_state.model_loader.get_model_summary()
        st.info(f"Features: {summary.get('n_features', 0)}")

    # Load model button
    if st.button("Load/Reload Model", type="primary", use_container_width=True):
        with st.spinner("Loading model..."):
            if st.session_state.model_loader.load_model():
                st.session_state.models_loaded = True
                st.session_state.run_id = st.session_state.model_loader.run_id
                st.success("Model loaded successfully!")
                st.rerun()
            else:
                st.error("Failed to load model")

    st.markdown("---")

    # Threshold configuration
    threshold = st.slider(
        "Fraud Threshold",
        min_value=0.30,
        max_value=0.60,
        value=st.session_state.model_loader.threshold if st.session_state.models_loaded else 0.44,
        step=0.01,
        help="Probability threshold for classifying as fraud",
    )

    # Model selection (for future multi-model support)
    model_type = st.selectbox(
        "Model Type",
        ["Stacking Ensemble", "XGBoost Only", "LightGBM Only", "CatBoost Only"],
        help="Select the model to use for predictions",
    )

    st.markdown("---")

    # Help section
    with st.expander("Help & Information"):
        st.markdown(
            """
        ### How to Use

        1. **Load Model**: Click the button above to load the trained model
        2. **Choose Input**: Select CSV upload, manual entry, or sample data
        3. **Run Prediction**: Click the predict button to get results
        4. **Export Results**: Download predictions as CSV

        ### Risk Levels

        - **Low** (< 50%): Likely legitimate
        - **Medium** (50-80%): Review recommended
        - **High** (> 80%): Likely fraud

        ### Features

        The model uses 30 engineered features including:
        - Transaction amount and timing
        - Card information
        - Address and distance metrics
        - Velocity features (C1-C14)
        - Anonymous features (V-features)
        """
        )

# Main content
if st.session_state.models_loaded:
    # Input tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload CSV", "✏️ Manual Entry", "🎲 Sample Data"])

    input_data = None

    with tab1:
        st.markdown("### Upload Transaction Data")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="CSV should contain transaction features",
        )

        if uploaded_file is not None:
            input_data = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(input_data)} transactions")

            with st.expander("Data Preview", expanded=True):
                st.dataframe(input_data.head(10), use_container_width=True)

            # Basic validation
            feature_names = st.session_state.model_loader.feature_names
            missing_features = [f for f in feature_names if f not in input_data.columns]

            if missing_features:
                st.warning(
                    f"Missing {len(missing_features)} features: {missing_features[:5]}... "
                    "(will use default values)"
                )

    with tab2:
        st.markdown("### Enter Transaction Details")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Transaction Info")
            transaction_amt = st.number_input(
                "Transaction Amount ($)", min_value=0.01, value=100.0, step=10.0
            )
            card1 = st.number_input("Card ID (card1)", min_value=1000, max_value=20000, value=10000)
            card2 = st.number_input("Card ID (card2)", min_value=100.0, max_value=600.0, value=300.0)
            card5 = st.number_input("Card5", min_value=100, max_value=300, value=200)

        with col2:
            st.subheader("Additional Features")
            addr1 = st.number_input("Address Code (addr1)", min_value=0, max_value=500, value=200)
            dist1 = st.number_input("Distance (dist1)", min_value=0.0, max_value=10000.0, value=50.0)
            c1 = st.number_input("C1 (Count)", min_value=0.0, max_value=100.0, value=1.0)
            c14 = st.number_input("C14 (Count)", min_value=0.0, max_value=100.0, value=1.0)

        if st.button("Create Transaction", key="manual_btn"):
            manual_transaction = {
                "TransactionAmt": transaction_amt,
                "card1": card1,
                "card2": card2,
                "card5": card5,
                "addr1": addr1,
                "dist1": dist1,
                "C1": c1,
                "C14": c14,
                "TransactionDT": 86400,  # Default
            }
            input_data = pd.DataFrame([manual_transaction])
            st.success("Transaction created")

    with tab3:
        st.markdown("### Generate Sample Data")

        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.number_input(
                "Number of Transactions", min_value=10, max_value=1000, value=100
            )
        with col2:
            fraud_rate = st.slider("Expected Fraud Rate (%)", min_value=1, max_value=20, value=4)

        if st.button("Generate Sample Data", key="sample_btn"):
            input_data = st.session_state.predictor.generate_sample_data(
                n_samples=n_samples,
                fraud_rate=fraud_rate / 100,
            )
            st.success(f"Generated {len(input_data)} sample transactions")

            with st.expander("Sample Data Preview"):
                st.dataframe(input_data.head(10), use_container_width=True)

    # Prediction section
    if input_data is not None:
        st.markdown("---")
        st.header("Run Prediction")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🔮 Predict Fraud", type="primary", use_container_width=True, key="run_prediction"
            ):
                with st.spinner("Analyzing transactions..."):
                    results = st.session_state.predictor.predict_batch(input_data, threshold=threshold)

                    if results["success"]:
                        st.success("Prediction complete!")

                        # Summary metrics
                        st.markdown("### Summary")
                        summary = results["summary"]

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Transactions", summary["total_count"])
                        with col2:
                            st.metric(
                                "Fraud Detected",
                                summary["fraud_count"],
                                delta=f"{summary['fraud_rate']*100:.1f}%",
                                delta_color="inverse",
                            )
                        with col3:
                            st.metric("Avg Probability", f"{summary['avg_probability']*100:.1f}%")
                        with col4:
                            st.metric("High Risk", summary["high_risk_count"])

                        # Risk distribution
                        st.markdown("### Risk Distribution")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Pie chart
                            risk_data = {
                                "Risk Level": ["High Risk", "Medium Risk", "Low Risk"],
                                "Count": [
                                    summary["high_risk_count"],
                                    summary["medium_risk_count"],
                                    summary["low_risk_count"],
                                ],
                            }
                            fig_pie = px.pie(
                                risk_data,
                                values="Count",
                                names="Risk Level",
                                color="Risk Level",
                                color_discrete_map={
                                    "High Risk": "#FF4B4B",
                                    "Medium Risk": "#FFA500",
                                    "Low Risk": "#00CC00",
                                },
                                title="Transaction Risk Distribution",
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                        with col2:
                            # Histogram
                            predictions_df = results["predictions"]
                            fig_hist = px.histogram(
                                predictions_df,
                                x="fraud_probability",
                                nbins=30,
                                title="Fraud Probability Distribution",
                                labels={"fraud_probability": "Fraud Probability"},
                                color_discrete_sequence=["#636EFA"],
                            )
                            fig_hist.add_vline(
                                x=threshold,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"Threshold: {threshold}",
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

                        # Detailed results
                        st.markdown("### Detailed Results")

                        # Sort by fraud probability descending
                        display_df = predictions_df.sort_values("fraud_probability", ascending=False)

                        # Color-code the risk level
                        def highlight_risk(row):
                            if row["risk_level"] == "high":
                                return ["background-color: #FFE4E4"] * len(row)
                            elif row["risk_level"] == "medium":
                                return ["background-color: #FFF4E4"] * len(row)
                            return [""] * len(row)

                        st.dataframe(
                            display_df[
                                ["transaction_id", "fraud_probability", "is_fraud", "risk_level"]
                            ]
                            if "transaction_id" in display_df.columns
                            else display_df[["fraud_probability", "is_fraud", "risk_level"]].head(
                                100
                            ),
                            use_container_width=True,
                        )

                        # Export section
                        st.markdown("### Export Results")

                        col1, col2 = st.columns(2)
                        with col1:
                            csv = predictions_df.to_csv(index=False)
                            st.download_button(
                                label="Download Full Results (CSV)",
                                data=csv,
                                file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )

                        with col2:
                            # Fraud only export
                            fraud_df = predictions_df[predictions_df["is_fraud"]]
                            if len(fraud_df) > 0:
                                fraud_csv = fraud_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Fraud Cases Only (CSV)",
                                    data=fraud_csv,
                                    file_name=f"fraud_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                )

                    else:
                        st.error(f"Prediction failed: {results['error']}")

else:
    # No model loaded state
    st.warning("Please load a model using the sidebar before making predictions.")

    with st.expander("First Time Setup", expanded=True):
        st.markdown(
            """
        ### Getting Started

        If this is your first time using the system:

        1. **Train a Model** (if not already done):
           ```bash
           python pipelines/training_pipeline.py --config-path config/params_codespaces.yaml
           ```

        2. **Click "Load/Reload Model"** in the sidebar

        3. **Start Making Predictions!**

        ### Service URLs

        - **MLflow UI**: [http://localhost:5000](http://localhost:5000) - View experiments
        - **MinIO Console**: [http://localhost:9001](http://localhost:9001) - View artifacts
          - Username: `minioadmin`
          - Password: `minioadmin`
        - **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) - REST API
        """
        )

# Footer
st.markdown("---")
st.caption("Fraud Detection MLOps Platform | Built with Streamlit")
