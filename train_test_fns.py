import torch
import torch.nn as nn
import torch.optim as optim

"""
Generic functions for training and testing a model. Intended to 
be agnostic to the structure of the model.
"""

def train_model(model, training_loader, loss_fn_cls=nn.MSELoss,
                optimizer_cls=optim.Adam, epochs=10, device="cpu"):

    loss_function = loss_fn_cls()
    optimizer = optimizer_cls(model.parameters())

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0
        for essays, scores in training_loader:

            essays = essays.float().to(device)
            scores = scores.float().to(device)

            # run any preprocessing the model specifies
            if getattr(model, "preprocess_fn", None) is not None:
                essays = model.preprocess_fn(essays)

            optimizer.zero_grad()

            scores_predicted = model(essays)
            scores_predicted = torch.squeeze(scores_predicted, 1)

            loss = loss_function(scores_predicted, scores)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"loss for epoch {epoch}: {epoch_loss}")

    return model


def test_model(model, test_loader, loss_fn=torch.nn.functional.mse_loss,
               device="cpu"):

    model.eval()
    with torch.no_grad():

        # total loss is the mse loss. total error is the total
        # of mispredictions (score - score_predicted), so we
        # can compute the average misprediction and use that as
        # a benchmark
        total_loss = 0
        total_error = 0

        for essays, scores in test_loader:

            essays = essays.float().to(device)
            scores = scores.float().to(device)

            # run any preprocessing the model specifies
            if getattr(model, "preprocess_fn", None) is not None:
                essays = model.preprocess_fn(essays)

            scores_predicted = model(essays)
            scores_predicted = torch.squeeze(scores_predicted, 1)
            loss = loss_fn(scores_predicted, scores)

            total_error += torch.sum(torch.abs(scores_predicted - scores))
            total_loss += loss

        print(f"total loss: {total_loss}")
        print(f"average error: {total_error / (len(test_loader)*test_loader.batch_size)}")