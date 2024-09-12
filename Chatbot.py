import pickle
import torch
import torch.nn as nn
import numpy as np


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
        self.responses = []
        self.model = None
        self.encoder = None
        self.load_model()

    def load_model(self, filename="chatbot_model.pkl"):
        try:
            with open(filename, "rb") as f:
                self.responses, _, state_dict = pickle.load(f)
                self.model = self.build_model()
                self.model.load_state_dict(state_dict)
        except FileNotFoundError:
            print("Model file not found. Please ensure the model is trained and saved.")

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

    def get_response(self, prompt):
        # Predict the response using the trained model
        prompt_encoded = self.encode_responses([prompt])
        prompt_encoded = self.pad_sequences(prompt_encoded, maxlen=100)
        prompt_encoded = torch.tensor(prompt_encoded, dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            predicted_index = torch.argmax(self.model(prompt_encoded)).item()
        return self.responses[predicted_index]


def main():
    bot = Chatbot()
    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            break

        response = bot.get_response(prompt)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()
