import os
import torch
import json
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score
from models.AASIST import load_preprocess_AASIST, Model


def get_labels(labels_file):
    labels = {}
    with open(labels_file, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) >= 5:  # Assuming the line structure is consistent
                file_id = parts[1].strip()  # Extract the file_id
                label = parts[-1].strip()   # Extract the label (e.g., "spoof" or "bonafide")
                # Assign 0 for spoof, 1 for real/bonafide
                labels[file_id] = 0 if label == 'spoof' else 1
    return labels


labels_file = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
dataset_path = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_eval/flac'

test_labels = get_labels(labels_file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the AASIST model
with open("./models/AASIST.conf", "r") as f_json:
    config = json.loads(f_json.read())
model_config = config["model_config"]

feat_model = Model(model_config)
feat_model.load_state_dict(torch.load("./weights/AASIST.pth", map_location=device))
feat_model = feat_model.to(device)  # Move model to the appropriate device
feat_model.eval()  # Set the model to evaluation mode

prediction_function = torch.nn.Softmax(dim=-1)

# Track predictions and ground truth
all_probabilities = []
all_ground_truths = []
all_predictions = []

# Iterate over audio files in the dataset path
for file_name in os.listdir(dataset_path):
    if file_name.endswith('.flac'):
        file_path = os.path.join(dataset_path, file_name)
        file_id = os.path.splitext(file_name)[0]  # Get file_id without extension
        try:
            # Preprocess the audio
            processed_audio = load_preprocess_AASIST(file_path)
            processed_audio = processed_audio.to(device)  # Move to the same device as the model

            # Perform prediction
            with torch.no_grad():
                logits = feat_model(processed_audio)
                probabilities = prediction_function(logits[1])

            # Get ground truth label
            ground_truth = test_labels.get(file_id)
            if ground_truth is not None:
                # Save the probability of the positive class (bonafide = 1)
                all_probabilities.append(probabilities[0, 1].item())
                all_ground_truths.append(ground_truth)

                # Get predicted class (0 or 1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                all_predictions.append(predicted_class)

                # Print results
                print(f"File: {file_name}, Predicted: {predicted_class}, Ground Truth: {ground_truth}, "
                      f"Probabilities: {probabilities.cpu().numpy()}")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

# Calculate Accuracy
if all_ground_truths and all_predictions:
    accuracy = accuracy_score(all_ground_truths, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")

# Calculate EER
if all_ground_truths and all_probabilities:
    all_ground_truths = np.array(all_ground_truths)
    all_probabilities = np.array(all_probabilities)

    # Use sklearn's roc_curve to calculate false positive and true positive rates
    fpr, tpr, thresholds = roc_curve(all_ground_truths, all_probabilities)

    # Calculate False Rejection Rate (FRR = 1 - TPR)
    frr = 1 - tpr

    # Find the threshold where FAR = FRR
    eer_threshold_index = np.nanargmin(np.abs(frr - fpr))
    eer = (fpr[eer_threshold_index] + frr[eer_threshold_index]) / 2

    print(f"EER: {eer:.4f}, Threshold: {thresholds[eer_threshold_index]:.4f}")
else:
    print("Insufficient data for accuracy or EER calculation.")
