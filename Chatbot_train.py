import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import wandb # if you want to use wandb for logging data
import time


class ChatbotModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(ChatbotModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


class Chatbot:
    def __init__(self):
        self.responses = [
            "Ethical hacking involves identifying vulnerabilities in systems.",
            "Penetration testing is a key aspect of ethical hacking.",
            "Always get permission before performing any hacking activities.",
            "Use tools like Nmap and Wireshark for network scanning.",
            "Stay updated with the latest security patches.",
            "Report any vulnerabilities you find to the responsible parties.",
            "Social engineering is a common technique used in hacking.",
            "Use strong, unique passwords for different accounts.",
            "Two-factor authentication adds an extra layer of security.",
            "Regularly back up your data to prevent data loss.",
            "Keep your software and systems updated to avoid exploits.",
            "Understand the legal implications of hacking activities.",
            "Use encryption to protect sensitive data.",
            "Network segmentation can help limit the impact of a breach.",
            "Always follow ethical guidelines and best practices.",
        ]
        self.feedback = [5] * len(self.responses)
        self.encoder = LabelEncoder()
        self.model = self.build_model()
        self.train_model()

    def build_model(self):
        input_dim = 1000
        embedding_dim = 64
        hidden_dim = 64
        output_dim = len(self.responses)
        model = ChatbotModel(input_dim, embedding_dim, hidden_dim, output_dim)
        return model

    def encode_responses(self, responses):
        encoded_responses = []
        for response in responses:
            encoded_response = [ord(char) for char in response]
            encoded_responses.append(encoded_response)
        return encoded_responses

    def pad_sequences(self, sequences, maxlen):
        padded_sequences = np.zeros((len(sequences), maxlen), dtype=int)
        for i, seq in enumerate(sequences):
            if len(seq) > maxlen:
                padded_sequences[i, :maxlen] = seq[:maxlen]
            else:
                padded_sequences[i, : len(seq)] = seq
        return padded_sequences

    def train_model(self):

        X = self.encode_responses(self.responses)
        y = self.encoder.fit_transform(range(len(self.responses)))
        X = self.pad_sequences(X, maxlen=100)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = torch.tensor(X_train, dtype=torch.long)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training
        for epoch in range(3000):
            start_time = time.time()
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test)
                val_loss = criterion(val_outputs, y_test)

            end_time = time.time()
            epoch_time = end_time - start_time

            epoch_time_milliseconds = epoch_time * 1000
            # Val Loss: Measures the model's performance on the validation data, indicating how well the model generalizes to new data. The goal is to minimize this value.
            # Loss: Measures the model's performance on the training data. The goal is to minimize this value.

            print(f"Epoch {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    def get_response(self, prompt):
        # Response preciction logic 
        prompt_encoded = self.encode_responses([prompt])
        prompt_encoded = self.pad_sequences(prompt_encoded, maxlen=100)
        prompt_encoded = torch.tensor(prompt_encoded, dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            predicted_index = torch.argmax(self.model(prompt_encoded)).item()
        return self.responses[predicted_index]

    def update_feedback(self, response, score, custom_response=None):
        if custom_response:
            if response in self.responses:
                index = self.responses.index(response)
                self.responses[index] = custom_response
            else:
                self.responses.append(custom_response)
                self.feedback.append(score)
        else:
            if response not in self.responses:
                self.responses.append(response)
                self.feedback.append(score)
            else:
                index = self.responses.index(response)
                self.feedback[index] = score

        self.model = self.build_model()
        self.train_model()
        self.save_model() 

    def save_model(self, filename="chatbot_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump((self.responses, self.feedback, self.model.state_dict()), f)

    def load_model(self, filename="chatbot_model.pkl"):
        try:
            with open(filename, "rb") as f:
                self.responses, self.feedback, state_dict = pickle.load(f)
                self.model = (
                    self.build_model()
                ) 
                self.model.load_state_dict(state_dict)
        except FileNotFoundError:
            pass 


def critic_feedback(response):
    keywords = [
        "ethical hacking",
        "penetration testing",
        "network scanning",
        "Nmap",
        "Wireshark",
        "security patches",
        "vulnerabilities",
        "social engineering",
        "strong passwords",
        "two-factor authentication",
        "data backup",
        "software updates",
        "legal implications",
        "encryption",
        "network segmentation",
        "ethical guidelines",
        "best practices",
        "cybersecurity",
        "firewall",
        "intrusion detection",
        "malware analysis",
        "phishing",
        "incident response",
        "threat intelligence",
        "vulnerability assessment",
    ]
    score = 5

    # Increase score for each keyword found
    for keyword in keywords:
        if keyword.lower() in response.lower():
            score += 1

    # Adjust score based on length of response
    if len(response) > 50:
        score += 1
    elif len(response) < 20:
        score -= 1

    return max(1, min(10, score))


def main():
    bot = Chatbot()
    bot.load_model()  # Load existing model if available

    interaction_count = 0
    critic_multiplier = 1.2

    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            break

        response = bot.get_response(prompt)
        print(f"Bot: {response}")

        try:
            score = int(input("Rate the response (1-10): "))
            if 1 <= score <= 10:
                custom_response = input(
                    "Enter the correct response (or press Enter to skip): "
                ).strip()
                if custom_response:
                    bot.update_feedback(response, score, custom_response)
                else:
                    bot.update_feedback(response, score)
                
                num_critic_feedbacks = int(critic_multiplier**interaction_count)
                for _ in range(num_critic_feedbacks):
                    critic_score = critic_feedback(response)
                    bot.update_feedback(response, critic_score)
                interaction_count += 1
            else:
                print("Please enter a score between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 10.")


if __name__ == "__main__":
    main()
