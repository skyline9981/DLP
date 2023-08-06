import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import os
import numpy as np
import matplotlib.pyplot as plt
from ResNet import ResNet
from dataloader import LeukemiaLoader
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from plot import PlotResult


def test():
    """
    testing process
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataset = LeukemiaLoader("", "test")
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False
    )
    model = ResNet.ResNet18(num_classes=2).to(device)  # type: ignore
    model.load_state_dict(torch.load("weight/Resnet18_best.pt"))
    model.eval()
    predict_result = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            predict_result.extend(torch.argmax(outputs, dim=1).tolist())

    save_result("resnet_18_test.csv", predict_result)

    # print("test() not defined")


def train():
    """
    training process
    you can use resnet18 resnet50 or resnet152 to train and compare the result
    """
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    num_epoch = 25
    num_classes: int = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    train_dataset = LeukemiaLoader("", "train")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    valid_dataset = LeukemiaLoader("", "valid")
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=True
    )

    # Model
    model = ResNet.ResNet18(num_classes=num_classes).to(device)  # type: ignore
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.5, 1.0])).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    accuracy = []
    loss_record = []
    # Training
    for epoch in range(num_epoch):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())
            if (i + 1) % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1,
                        num_epoch,
                        i + 1,
                        len(train_loader),
                        loss.item(),
                    )
                )

        # Validation
        model.eval()
        ground_truth = []
        predict_result = []
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                predict_result.extend(torch.argmax(outputs, dim=1).tolist())
                ground_truth.extend(labels.tolist())
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        accuracy.append(acc)

        print("Accuracy of the model on the valid images: {} %".format(acc))
        if acc >= max(accuracy):  # type: ignore
            path = "weight"
            # Check whether the specified path exists or not
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            # save model
            torch.save(
                model.state_dict(),
                "weight/" + "Resnet18_best" + ".pt",
            )
            # torch.save(
            #     model.state_dict(),
            #     "weight/" + "Resnet50_best" + ".pt",
            # )
            # torch.save(
            #     model.state_dict(),
            #     "weight/" + "Resnet152_best" + ".pt",
            # )

        # Confusion Matrix
        cm = ConfusionMatrixDisplay.from_predictions(ground_truth, predict_result)
        # plt.show()
        cm.plot()
        record_path = "record"
        isExist = os.path.exists(record_path)
        if not isExist:
            os.makedirs(record_path)
        cm.figure_.savefig(f"./record/cm_{epoch}.png")

    # print max accuracy
    print("Max accuracy: ", max(accuracy))

    path = "record"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    with open("record/" + "ResNet18" + ".txt", "w") as f:
        for i in accuracy:
            f.write(str(i) + "\n")

    with open("record/" + "ResNet18_loss" + ".txt", "w") as f:
        for i in loss_record:
            f.write(str(i) + "\n")

    # print("train() not defined")


def evaluate(model_name: str):
    """
    evaluate process
    you can use resnet18 resnet50 or resnet152 to evaluate and compare the result
    """
    # Hyperparameters
    batch_size = 16
    num_classes: int = 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    valid_dataset = LeukemiaLoader("", "valid")
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=True
    )

    if model_name == "ResNet18":
        model = ResNet.ResNet18(num_classes=num_classes).to(device)  # type: ignore
    elif model_name == "ResNet50":
        model = ResNet.ResNet50(num_classes=num_classes).to(device)  # type: ignore
    elif model_name == "ResNet152":
        model = ResNet.ResNet152(num_classes=num_classes).to(device)  # type: ignore
    else:
        print("model name error")
        return

    weight_path = "weight/" + model_name + "_best.pt"
    model.load_state_dict(torch.load(weight_path))

    model.eval()
    ground_truth = []
    predict_result = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predict_result.extend(torch.argmax(outputs, dim=1).tolist())
            ground_truth.extend(labels.tolist())
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total

    print("Accuracy of the {} on the valid images: {} %".format(model_name, acc))

    # Confusion Matrix
    cm = ConfusionMatrixDisplay.from_predictions(ground_truth, predict_result)
    # plt.show()


def save_result(csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df["ID"] = df["Path"]
    new_df["label"] = predict_result
    new_df.to_csv("./312554012_resnet18.csv", index=False)


if __name__ == "__main__":
    # print("Good Luck :)")

    # train()

    evaluate("ResNet18")
    evaluate("ResNet50")
    evaluate("ResNet152")
    # test()

    # plotObject = PlotResult("ResNet50", 25)
    # plotObject.plot_compare("ResNet18", "ResNet50", "ResNet152")
    # plotObject.plot()
    # plotObject.plot_loss()
