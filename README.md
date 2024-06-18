## Email Spam Classification

This project classifies emails as spam or not spam using a fine-tuned DistilBERT model. The project includes data preprocessing, model training, evaluation, and real-time email classification.

### Table of Contents
1. [Installation](#installation)
2. [Setup Kaggle Dataset](#setup-kaggle-dataset)
3. [Training the Model](#training-the-model)
4. [Evaluating the Model](#evaluating-the-model)
5. [Real-time Email Classification](#real-time-email-classification)
6. [Translation](#translation)
7. [Results](#results)

### Installation

To get started with this project, you need to set up your environment with the required libraries. Follow the steps below:

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repository/Spam_Detection.git
    cd Spam_Detection
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Setup Kaggle Dataset

1. Make sure you have a Kaggle account and have generated an API token. Place the `kaggle.json` file in your home directory under `.kaggle/`.

2. Run the `setup_kaggle.py` script to download and prepare the dataset:
    ```sh
    python setup_kaggle.py
    ```

### Training the Model

1. To train the model using the dataset, run the `main_script.py`:
    ```sh
    python main_script.py
    ```

This script will:
- Load and preprocess the dataset.
- Split the data for training.
- Train the DistilBERT model.
- Save the trained model to `./trained_model`.

### Evaluating the Model

1. To evaluate the trained model on the remaining 80% of the dataset, run the `evaluation.py` script:
    ```sh
    python evaluation.py
    ```

This script will:
- Load the remaining dataset.
- Evaluate the model.
- Print the accuracy, precision, recall, and F1 score.

### Real-time Email Classification

1. Create a file named `requirements.py` and update it with your Gmail credentials:
    ```python
    EMAIL = 'your-email@gmail.com'
    PASSWORD = 'your-app-password'
    ```

2. Run the `predict.py` script to classify real-time emails:
    ```sh
    python predict.py
    ```

This script will:
- Connect to your Gmail account.
- Fetch unread emails.
- Translate the email content to English.
- Classify the email as spam or not spam.
- Label the email as "Potential Spam" if classified as spam.

### Translation

The `predict.py` script includes functionality to translate email content to English using Google Translate API.

### Results

Results of the evaluation and real-time classification will be printed to the console, including accuracy, precision, recall, and F1 score for evaluation, and classification results for real-time emails.
