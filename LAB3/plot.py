import matplotlib.pyplot as plt


class PlotResult:
    def __init__(self, model_name, epoch_num):
        self.model_name = model_name
        self.epoch_num = epoch_num

    def plot(self):
        epoch = [i for i in range(self.epoch_num)]
        accuracy = []

        with open("record/" + self.model_name + ".txt", "r") as f:
            for line in f.readlines():
                s = line.strip("\n")
                accuracy.append(float(s))

        title_name = "Result comparaison (" + self.model_name + ")"
        plt.title(title_name, fontsize=18)
        # plt.plot(epoch, acc, 'C0o-', linewidth=1, markersize=2, label="xxx")
        plt.plot(epoch, accuracy, "-", linewidth=2, label=self.model_name)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy(%)", fontsize=12)
        plt.legend(loc="upper left", fontsize=9)
        plt.savefig("record/" + self.model_name + "_accuracy.png")
        plt.show()

    def plot_compare(self, model_name1, model_name2, model_name3):
        epoch = [i for i in range(self.epoch_num)]
        accuracy1 = []
        accuracy2 = []
        accuracy3 = []

        with open("record/" + model_name1 + ".txt", "r") as f:
            for line in f.readlines():
                s = line.strip("\n")
                accuracy1.append(float(s))

        with open("record/" + model_name2 + ".txt", "r") as f:
            for line in f.readlines():
                s = line.strip("\n")
                accuracy2.append(float(s))

        with open("record/" + model_name3 + ".txt", "r") as f:
            for line in f.readlines():
                s = line.strip("\n")
                accuracy3.append(float(s))

        title_name = (
            "Result comparaison ("
            + model_name1
            + ", "
            + model_name2
            + ", "
            + model_name3
            + ")"
        )
        plt.title(title_name, fontsize=14)
        # plt.plot(epoch, acc, 'C0o-', linewidth=1, markersize=2, label="xxx")
        plt.plot(epoch, accuracy1, "-", linewidth=2, label=model_name1, color="red")
        plt.plot(epoch, accuracy2, "-", linewidth=2, label=model_name2, color="blue")
        plt.plot(epoch, accuracy3, "-", linewidth=2, label=model_name3, color="green")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy(%)", fontsize=12)
        plt.legend(loc="upper left", fontsize=9)
        plt.savefig(
            "record/"
            + model_name1
            + "_"
            + model_name2
            + "_"
            + model_name3
            + "_accuracy.png"
        )
        plt.show()

    def plot_loss(self):
        loss = []

        with open("record/" + self.model_name + "_loss.txt", "r") as f:
            for line in f.readlines():
                s = line.strip("\n")
                loss.append(float(s))

        step = [len(loss)]

        title_name = "Result comparaison (" + self.model_name + ")"
        plt.title(title_name, fontsize=18)
        # plt.plot(epoch, acc, 'C0o-', linewidth=1, markersize=2, label="xxx")
        plt.plot(step, loss, "-", linewidth=2, label=self.model_name)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(loc="upper left", fontsize=9)
        plt.savefig("record/" + self.model_name + "_loss.png")
        plt.show()
