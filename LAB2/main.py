import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import torch.utils.data
from EEG import EEGNet
from DeepConv import DeepConvNet
from ShallowConv import ShallowConvNet
from dataloader import read_bci_data
from plot import plotResult

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="EEGNet")
parser.add_argument("-s", "--save", type=str, nargs="+")
parser.add_argument("-o", "--others", type=str, nargs="+")
args = parser.parse_args()

model_name = args.model
epoch_num = 500


if args.others != None and "draw" in args.others:
    plotObj = plotResult(model_name=model_name, epoch_num=epoch_num)
    plotObj.plot()

else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_data, train_label, test_data, test_label = read_bci_data()
    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)

    train_tensor = torch.utils.data.TensorDataset(train_data, train_label)
    test_tensor = torch.utils.data.TensorDataset(test_data, test_label)

    activation_func = [nn.ReLU(), nn.LeakyReLU(), nn.ELU()]
    func = ["ReLU", "LeakyReLU", "ELU"]

    acc = []

    for act, func_name in zip(activation_func, func):
        if func_name == "ReLU":
            print("\nUsing ReLU ...")
        elif func_name == "LeakyReLU":
            print("\nUsing LeakyReLU ...")
        elif func_name == "ELU":
            print("\nUsing ELU ...")

        if args.others != None and "time" in args.others:
            # record cuda time of training & testing
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()  # type: ignore

        if model_name == "EEGNet":
            model = EEGNet(device=device, activation_func=act)
        elif model_name == "DeepConvNet":
            model = DeepConvNet(device=device, activation_func=act)
        elif model_name == "ShallowConvNet":
            model = ShallowConvNet(device=device, activation_func=act)
        else:
            print("Model name error.")
            exit()

        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_tensor, batch_size=64, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_tensor, batch_size=64, shuffle=True
        )

        if args.others != None and "test" in args.others:
            model.load_state_dict(
                torch.load("weight/" + model_name + "_" + func_name + ".pt")
            )
            # testing process
            total_test = 0
            correct_test = 0
            model.eval()
            for i, (data, label) in enumerate(test_loader):
                data = data.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.long)

                # no need to calculate gradient and loss function

                # forward propagation
                output = model(data)

                # get predictions from the maximum value
                prediction = torch.max(output.data, 1)[1]

                # total number of labels
                total_test += len(label)

                # total correct predictions
                correct_test += (prediction == label).float().sum()

            # calculate accuracy
            max_accuracy = 100 * (correct_test / total_test)
            acc.append(max_accuracy.item())  # type: ignore

        else:
            accuracy_train = []
            accuracy_test = []
            for epoch in range(epoch_num):
                # training process
                total_loss = 0
                total_train = 0
                correct_train = 0
                model.train()
                for i, (data, label) in enumerate(train_loader):
                    data = data.to(device, dtype=torch.float)
                    label = label.to(device, dtype=torch.long)

                    # clear gradient
                    optimizer.zero_grad()

                    # forward propagation
                    output = model(data)

                    # calculate cross entropy (loss function)
                    loss = criterion(output, label)
                    total_loss += loss

                    # get predictions from the maximum value
                    prediction = torch.max(output.data, 1)[1]

                    # total number of labels
                    total_train += len(label)

                    # total correct predictions
                    correct_train += (prediction == label).float().sum()

                    # Calculate gradients
                    loss.backward()

                    # Update parameters
                    optimizer.step()

                # calculate accuracy
                accuracy = 100 * (correct_train / total_train)
                accuracy_train.append(accuracy.item())  # type: ignore

                if epoch % 100 == 99:
                    print("\nepoch ", epoch + 1, ":")
                    print("trainig accuracy: ", accuracy, "  loss: ", total_loss)

                # testing process
                total_test = 0
                correct_test = 0
                model.eval()
                for i, (data, label) in enumerate(test_loader):
                    data = data.to(device, dtype=torch.float)
                    label = label.to(device, dtype=torch.long)

                    # no need to calculate gradient and loss function

                    # forward propagation
                    output = model(data)

                    # get predictions from the maximum value
                    prediction = torch.max(output.data, 1)[1]

                    # total number of labels
                    total_test += len(label)

                    # total correct predictions
                    correct_test += (prediction == label).float().sum()

                # calculate accuracy
                accuracy = 100 * (correct_test / total_test)
                accuracy_test.append(accuracy.item())  # type: ignore

                if epoch % 100 == 99:
                    print("testing accuracy: ", accuracy)

            max_accuracy = max(accuracy_test)
            print(
                "\n"
                + func_name
                + " has max accuracy "
                + str(max_accuracy)
                + "% at epoch "
                + str(accuracy_test.index(max_accuracy))
            )
            acc.append(max_accuracy)

            if args.others != None and "time" in args.others:
                # print execution time
                end.record()  # type: ignore
                torch.cuda.synchronize()
                print("execution time: " + str(start.elapsed_time(end) / 1000) + "s")  # type: ignore

            if args.save != None and "record" in args.save:
                path = "record"
                # Check whether the specified path exists or not
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                with open(
                    "record/" + model_name + "_" + func_name + "_train.txt", "w"
                ) as f:
                    for i in accuracy_train:
                        f.write(str(i) + "\n")
                with open(
                    "record/" + model_name + "_" + func_name + "_test.txt", "w"
                ) as f:
                    for i in accuracy_test:
                        f.write(str(i) + "\n")
            if args.save != None and "weight" in args.save:
                path = "weight"
                # Check whether the specified path exists or not
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                # save model
                torch.save(
                    model.state_dict(), "weight/" + model_name + "_" + func_name + ".pt"
                )

        print(
            "Accuracy of {} with {} activation function: {}".format(
                model_name, func_name, acc[-1]
            )
        )
