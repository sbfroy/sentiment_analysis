import torch

train_class_counts = {
    -1.0: 379,
    -0.6: 2287,
    -0.2: 6004,
     0.2: 11304,
     0.6: 12614,
     1.0: 2161
}

total_train_samples = sum(train_class_counts.values())
class_weights = {interval: total_train_samples / count for interval, count in train_class_counts.items()}
normalized_class_weights = {interval: weight / len(train_class_counts) for interval, weight in class_weights.items()}

tensors = torch.tensor(list(normalized_class_weights.values()))

def get_weight(scaled_score):
    rounded_score = round(scaled_score, 1) # because floating point uncertainty
    return normalized_class_weights[rounded_score]
