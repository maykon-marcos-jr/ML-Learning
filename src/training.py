
import torch
print("PyTorch Version:", torch.__version__)
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
# progress bar
from tqdm.auto import tqdm # to automatically guess the environment
# To verify the speed performance
from timeit import default_timer as timer
import matplotlib.pyplot as plt

RND_SEED = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

print("Running PyTorch on:", device)

import requests
from pathlib import Path

if Path("src/helper_functions.py").is_file():
    print("request satisfied")
else:
    req = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("src/helper_functions.py", "wb") as f:
        f.write(req.content)

def print_train_timer(start: float,
                      end: float,
                      device: torch.device = None
                      ) -> float:
    """
    prints the difference between start and end
    """
    total_time = end - start
    print(f"Total time on {device}: {total_time:.3f} seconds")
    return total_time


def plot_loss_curves(results: dict[str, list[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_eval": [...],
             "test_loss": [...],
             "test_eval": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results["train_loss"]
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_eval']
    test_accuracy = results['test_eval']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_evaluation')
    plt.plot(epochs, test_accuracy, label='test_evaluation')
    plt.title('Evaluation')
    plt.xlabel('Epochs')
    plt.legend();


def train_step(
        model: nn.Module,
        X: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        is_classification: bool=False,
        eval_fn=None
    ) -> tuple[Tensor, Tensor|None]:
    """
    Execute a simple training step on the model based on the data
        Wheter the data is a batch or a full dataset
    It uses the loss function to guide the optimizer
    It `does not` set the model to train mode

    *** Args
    * `X`: The training data
    * `y`: The labels corresponding to the training data
    * `loss_fn`: Measures of predictions error compared with the labels `y`
        must receive the prediction and the labels in this specific order
    * `optimizer`: The optimizer used to calibrate the model parameters
        must be previously set to work with the model parameter
    * `is_classification`: Wheter is a classification mode or not
        If it is, is assumed that the `eval_fn` will require the use of `torch.argmax(dim=1)` to be applied on the predictions
    * `eval_fn`: Optional.
        A different function to measure the model current performance
        Must receive the parameters in the Scikit Learn order (that is, labels/y first, then the predictions)
        Must return a torch.Tensor-compatible value

    *** Returns
    * `loss`: the loss value
    * `eval`: the custom evaluation value, None if eval_fn is None
    """

    # Do the Forward Pass
    preds = model(X)

    # Calculate Loss
    loss = loss_fn(preds, y)

    # Optimizer Zero Grad
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Gradient Descent
    optimizer.step()

    eval = None
    if not eval_fn is None:
        p = preds.argmax(dim=1) if is_classification else preds
        eval = torch.tensor(eval_fn(y, p))

    return loss, eval


def test_step(
        model: nn.Module,
        X: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
        is_classification: bool = False,
        eval_fn = None
    ) -> tuple[Tensor, Tensor|None]:
    """
    Execute a simple testing step on the model based on the data
        Wheter the data is a batch or a full dataset
    It `does not` set the model to eval mode nor inference mode

    *** Args
    * `X`: The testing data
    * `y`: The labels corresponding to the testing data
    * `loss_fn`: Measures of predictions error compared with the labels `y`
        must receive the prediction and the labels in this specific order
    * `is_classification`: Wheter is a classification mode or not
        If it is, is assumed that the `eval_fn` will require the use of `torch.argmax(dim=1)` to be applied on the predictions
    * `eval_fn`: Optional.
        A different function to measure the model current performance
        Must receive the parameters in the Scikit Learn order (that is, labels/y first, then the predictions)
        Must return a torch.Tensor-compatible value

    *** Returns
    * `loss`: the loss value
    * `eval`: the custom evaluation value, None if eval_fn is None
    """

    # Do the Forward Pass
    preds = model(X)

    # Calculate Loss
    loss = loss_fn(preds, y)

    eval = None
    if not eval_fn is None:
        p = preds.softmax(dim=1).argmax(dim=1) if is_classification else preds
        eval = torch.tensor(eval_fn(y, p))

    return loss, eval


def training_loop_batch(
        model: nn.Module,
        data: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        eval_fn=None,
        is_classification: bool=False,
    ) -> tuple[Tensor, Tensor|None]:
    """
    Performs a epoch of model training
    Iterating through all the dataloader in batches

    *** Args
    * `data`: The dataloader contaning the data in batches
    * `loss_fn`: Measures of predictions error compared with the labels `y`
        must receive the prediction and the labels in this specific order
    * `optimizer`: The optimizer used to calibrate the model parameters
        must be previously set to work with the model parameter
    * `is_classification`: Wheter is a classification mode or not
        If it is, is assumed that the `eval_fn` will require the use of `torch.argmax(dim=1)` to be applied on the predictions
    * `eval_fn`: Optional.
        A different function to measure the model current performance
        Must receive the parameters in the Scikit Learn order (that is, labels/y first, then the predictions)
        Must return a torch.Tensor-compatible value

    *** Returns
    * `loss`: the loss value
    * `eval`: the custom evaluation value, None if eval_fn is None
    """

    # Sets the model to training mode
    model.train()
    accumulate_loss = torch.tensor(0).float().to(device)
    accumulate_eval = torch.tensor(0).float().to(device)
    size = torch.tensor(len(data)).float().to(device)

    for (X, y) in data:
        X = X.to(device)
        y = y.to(device)
        # performs a step
        loss, eval = train_step(
            model, X, y, loss_fn, optimizer,
            is_classification, eval_fn
        )
        accumulate_loss += loss

        if not eval is None:
            accumulate_eval += eval


    accumulate_loss /= size
    if eval_fn is None:
        accumulate_eval = None
    else:
        accumulate_eval /= size

    return accumulate_loss, accumulate_eval


def testing_loop_batch(
        model: nn.Module,
        data: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        eval_fn=None,
        is_classification: bool=False,
    ) -> tuple[Tensor, Tensor|None]:
    """
    Performs a epoch of model testing
    Iterating through all the dataloader in batches

    *** Args
    * `data`: The dataloader contaning the data in batches
    * `loss_fn`: Measures of predictions error compared with the labels `y`
        must receive the prediction and the labels in this specific order
    * `is_classification`: Wheter is a classification mode or not
        If it is, is assumed that the `eval_fn` will require the use of `torch.argmax(dim=1)` to be applied on the predictions
    * `eval_fn`: Optional.
        A different function to measure the model current performance
        Must receive the parameters in the Scikit Learn order (that is, labels/y first, then the predictions)
        Must return a torch.Tensor-compatible value

    *** Returns
    * `loss`: the loss value
    * `eval`: the custom evaluation value, None if eval_fn is None
    """

    # Sets the model to training mode
    model.eval()

    accumulate_loss = torch.tensor(0).float().to(device)
    accumulate_eval = torch.tensor(0).float().to(device)
    size = torch.tensor(len(data)).float().to(device)

    with torch.inference_mode():
        for (X, y) in data:
            X = X.to(device)
            y = y.to(device)
            # performs a step
            loss, eval = test_step(
                model, X, y, loss_fn,
                is_classification, eval_fn
            )
            accumulate_loss += loss

            if not eval is None:
                accumulate_eval += eval

        accumulate_loss /= size
        if eval_fn is None:
            accumulate_eval = None
        else:
            accumulate_eval /= size

    return accumulate_loss, accumulate_eval


def train_model_with_batches(
        model: nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        loss_fn: nn.Module,
        optimizer_fn,
        lr: float,
        N_EPOCHS: int,
        RND_SEED: int=None, # type: ignore
        eval_fn=None,
        is_classification: bool=False,
    ) -> dict:
    """
    Executes a full model training
    Iterating through all the dataloader in batches

    *** Args
    * `data`: The dataloader contaning the data in batches
    * `loss_fn`: Measures of predictions error compared with the labels `y`
        must receive the prediction and the labels in this specific order
    * `optimizer`: The optimizer module (non-instantiated) used to calibrate the model parameters
        Will be set to work with the model parameters
    * `lr`: Learning rate to be passed to the optimizer
    * `N_EPOCHS`: Number of training epochs
    * `RND_SEED`: Value for torch.manual_seed
    * `eval_fn`: Optional.
        A different function to measure the model current performance
        Must receive the parameters in the Scikit Learn order (that is, labels/y first, then the predictions)
        Must return a torch.Tensor-compatible value
    * `is_classification`: Wheter is a classification mode or not
        If it is, is assumed that the `eval_fn` will require the use of `torch.argmax(torch.softmax(dim=1), dim=1)` to be applied on the predictions
    """
    if not RND_SEED is None:
        torch.manual_seed(RND_SEED)

    optimizer = optimizer_fn(
        params=model.parameters(),
        lr=lr
    )

    results = {
        "train_loss": [],
        "train_eval": [],
        "test_loss": [],
        "test_eval": [],
        "training_time": 0.0,
    }

    device = next(model.parameters()).device

    # Start counting the time
    start = timer()

    for ep in tqdm(range(1, N_EPOCHS+1)):
        train_loss, train_eval = training_loop_batch(
            model,
            train_data,
            loss_fn,
            optimizer,
            device,
            eval_fn,
            is_classification,
        )
        test_loss, test_eval = testing_loop_batch(
            model,
            test_data,
            loss_fn,
            device,
            eval_fn,
            is_classification,
        )
        print(f"Epoch: {ep}")
        print(f" - Train Loss: {train_loss:.4f}")
        print(f" - Train Eval: {train_eval:.2f}%")
        print(f" - Test Loss:  {test_loss:.4f}")
        print(f" - Test Eval: {test_eval:.2f}%")
        results["train_loss"].append(train_loss.item())
        results["train_eval"].append(train_eval.item())
        results["test_loss"].append(test_loss.item())
        results["test_eval"].append(test_eval.item())

    training_time = print_train_timer(
        start,
        timer(),
        device,
    )
    results["training_time"] = training_time

    return results


def eval_model(
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        eval_fn=None,
        ) -> dict:
    loss, eval = testing_loop_batch(
        model, data_loader, loss_fn,
        device, eval_fn, True,
    )
    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_eval (%)": eval.item(),
        "device": str(device),
        }
