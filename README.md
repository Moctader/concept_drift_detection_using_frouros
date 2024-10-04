# Stock Data Pipeline with Concept Drift Detection

This project implements a stock data pipeline using yfinance for data fetching, scikit-learn for preprocessing, Keras for building and training an LSTM model, and frouros for concept drift detection. The configuration is managed using hydra and omegaconf.


## Installation

  1. Clone the repository:
       ```bash
        git clone https://github.com/Moctader/concept_drift_detection_using_frouros.git
        
        cd

  2. Create a virtual environment and activate it
        ```bash
        python3 -m venv venv
        source venv/bin/activate

  3. Install the required packages
        ```bash
        pip install -r requirements.txt


## Workflow

1. Initialize the Pipeline:

    Create an instance of StockDataPipeline with the specified stock symbol, split ratio, and sequence length.

    Fetch and preprocess the stock data.

    Build the LSTM model using the specified layers.

    Train the model on the historical data.

2. Initialize the Drift Detector:

    Create an instance of ConceptDriftDetector with the specified parameters.

3. Run the Stream Processor:

    Create an instance of StreamProcessor with the pipeline and detector.
    Run the stream test for concept drift detection on the current data.
    Detect Feature and Target Drift:

    Detect feature drift using the detect_feature_drift method.
    Detect target drift using the detect_target_drift method.

4. Plot the Results:

    Plot the true values, predicted values, and concept drift point using the plot_results function.