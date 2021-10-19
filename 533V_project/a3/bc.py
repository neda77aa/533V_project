import gym
import torch
from eval_policy import eval_policy, device
from model import MyModel
import torch.nn as nn
from dataset import Dataset

BATCH_SIZE = 64
TOTAL_EPOCHS = 10
LEARNING_RATE = 10e-4
PRINT_INTERVAL = 500
TEST_INTERVAL = 2

ENV_NAME = 'CartPole-v0'

dataset = Dataset(data_path="{}_dataset.pkl".format(ENV_NAME))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

env = gym.make(ENV_NAME)

# TODO INITIALIZE YOUR MODEL HERE
model = MyModel(4,2)
def initialize_weights(m):
     if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
#initializing the weights
model.apply(initialize_weights)

def train_behavioral_cloning():
    
    # TODO CHOOSE A OPTIMIZER AND A LOSS FUNCTION FOR TRAINING YOUR NETWORK
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    gradient_steps = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        for iteration, data in enumerate(dataloader):
            data = {k: v.to(device) for k, v in data.items()}

            output = model(data['state'].float())

            loss = loss_function(output, data["action"].argmax(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if gradient_steps % PRINT_INTERVAL == 0:
                print('[epoch {:4d}/{}] [iter {:7d}] [loss {:.5f}]'
                    .format(epoch, TOTAL_EPOCHS, gradient_steps, loss.item()))
            
            gradient_steps += 1

        if epoch % TEST_INTERVAL == 0:
            score = eval_policy(policy=model, env=ENV_NAME)
            print('[Test on environment] [epoch {}/{}] [score {:.2f}]'
                .format(epoch, TOTAL_EPOCHS, score))

    model_name = "behavioral_cloning_{}.pt".format(ENV_NAME)
    print('Saving model as {}'.format(model_name))
    torch.save(model.state_dict(), model_name)


if __name__ == "__main__":
    train_behavioral_cloning()
