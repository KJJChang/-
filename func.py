import torch


def train_step(dataloader, model, cost_fn, optimizer, accuracy_fn, device):
    train_cost = 0
    train_acc = 0
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        model.train()
        y_pred = model(x)
        cost = cost_fn(y_pred, y)
        train_cost += cost
        train_acc += accuracy_fn(y_pred.argmax(dim=1), y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    train_cost /= len(dataloader)
    train_acc /= len(dataloader)

    print(f"Train Cost: {train_cost:.4f}, Train Accuracy: {train_acc:.2f}")


def test_step(dataloader, model, cost_fn, accuracy_fn, device):
    test_cost = 0
    test_acc = 0
    model.eval()

    with torch.inference_mode():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            test_pred = model(x)
            cost = cost_fn(test_pred, y)
            test_cost += cost
            test_acc += accuracy_fn(test_pred.argmax(dim=1), y)
        test_cost /= len(dataloader)
        test_acc /= len(dataloader)

    print(f"Test Cost: {test_cost:.4f}, Test Accuracy: {test_acc:.2f}\n")


def accuracy(y_pred, y_true):
    correct_num = (y_pred == y_true).sum()
    acc = correct_num / len(y_true) * 100
    return acc