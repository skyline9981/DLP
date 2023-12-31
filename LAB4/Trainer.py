import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import (
    Generator,
    Gaussian_Predictor,
    Decoder_Fusion,
    Label_Encoder,
    RGB_Encoder,
)

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10


def Generate_PSNR(imgs1, imgs2, data_range=1.0):
    """PSNR for torch tensor"""
    imgs1 = imgs1.to(args.device).detach()
    imgs2 = imgs2.to(args.device).detach()
    mse = nn.functional.mse_loss(imgs1, imgs2)  # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    psnr = psnr.detach().cpu()
    return psnr


def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    return KLD


class kl_annealing:
    def __init__(self, args, current_epoch=0):
        super().__init__()
        self.use_cycle = args.kl_anneal_type == "Cyclical"
        self.n_iter = args.num_epoch
        self.n_cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.period = self.n_iter / self.n_cycle
        self.step = self.ratio / self.period
        self.current_epoch = current_epoch
        self.epoch = 0
        self.v = 0
        self.beta = args.beta

    def update(self):
        self.epoch += 1
        if self.use_cycle:
            if self.epoch % self.period == 0:
                self.beta = 0.0001
            else:
                self.beta += self.step
            if self.beta > 1.0:
                self.beta = 1.0
        else:
            self.v += self.step / 2
            if self.v > 1.0:
                self.v = 1.0
            self.beta = self.v

    def get_beta(self):
        return self.beta


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)

        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor = Gaussian_Predictor(
            args.F_dim + args.L_dim, args.N_dim
        )
        self.Decoder_Fusion = Decoder_Fusion(
            args.F_dim + args.L_dim + args.N_dim, args.D_out_dim
        )

        # Generative model
        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)

        self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=3, gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0

        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde

        self.train_vi_len = args.train_vi_len
        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size

    def forward(self, img, label):
        pass

    def training_stage(self):
        all_epoch_loss = []
        all_epoch_mse = []
        all_epoch_kld = []
        all_ave_psnr = []

        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False  # type: ignore

            epoch_mse = 0
            epoch_kld = 0
            epoch_loss = 0

            for img, label in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss, mse, kld = self.training_one_step(
                    img, label, adapt_TeacherForcing
                )
                epoch_mse += mse
                epoch_kld += kld
                epoch_loss += loss

                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar(
                        "train [TeacherForcing: ON, {:.1f}], beta: {}".format(
                            self.tfr, beta
                        ),
                        pbar,
                        loss.detach().cpu(),
                        lr=self.scheduler.get_last_lr()[0],
                    )
                else:
                    self.tqdm_bar(
                        "train [TeacherForcing: OFF, {:.1f}], beta: {}".format(
                            self.tfr, beta
                        ),
                        pbar,
                        loss.detach().cpu(),
                        lr=self.scheduler.get_last_lr()[0],
                    )

            if self.current_epoch % self.args.per_save == 0:
                self.save(
                    os.path.join(
                        self.args.save_root, f"epoch={self.current_epoch}.ckpt"
                    )
                )
            avg_mse = epoch_mse / self.train_vi_len
            avg_kld = epoch_kld / self.train_vi_len
            avg_loss = epoch_loss / self.train_vi_len
            avg_mse = avg_mse.detach().cpu()
            avg_kld = avg_kld.detach().cpu()
            avg_loss = avg_loss.detach().cpu()
            all_epoch_mse.append(avg_mse)
            all_epoch_kld.append(avg_kld)
            all_epoch_loss.append(avg_loss)

            with open("./{}/loss_record.txt".format(args.log_dir), "a") as f:
                f.write(f"{avg_loss:.5f}\n")

            with open("./{}/MSEloss_record.txt".format(args.log_dir), "a") as f:
                f.write(f"{avg_mse:.5f}\n")

            with open("./{}/KLDloss_record.txt".format(args.log_dir), "a") as f:
                f.write(f"{avg_kld:.5f}\n")

            psnr = self.eval()  # type: ignore

            with open("./{}/PSNR_record.txt".format(args.log_dir), "a") as f:
                f.write(f"{psnr:.5f}\n")  # type: ignore

            all_ave_psnr.append(psnr)  # type: ignore
            self.current_epoch += 1
            if self.current_epoch <= 7:
                self.scheduler.step()
            self.teacher_forcing_ratio_update()
            # self.kl_annealing.update()

    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        psnr = 0.0
        for img, label in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            psnr = self.val_one_step(img, label)
            self.tqdm_bar(
                "val", pbar, psnr, lr=self.scheduler.get_last_lr()[0]  # type: ignore
            )
        return psnr

    def training_one_step(self, img, label, adapt_TeacherForcing):
        img = img.permute(1, 0, 2, 3, 4)  # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4)  # change tensor into (seq, B, C, H, W)

        self.optim.zero_grad()

        h_seq = [self.frame_transformation(img[i]) for i in range(self.train_vi_len)]
        p_seq = [self.label_transformation(label[i]) for i in range(self.train_vi_len)]

        mse = 0
        kld = 0
        for t in range(1, self.train_vi_len):
            h_t = h_seq[t]
            h_t_1 = h_seq[t - 1]
            p_t = p_seq[t]

            z_t, mu, logvar = self.Gaussian_Predictor(h_t, p_t)
            g_t = self.Decoder_Fusion(h_t_1, p_t, z_t)
            img_gen = self.Generator(g_t)

            if not adapt_TeacherForcing:
                h_seq[t] = self.frame_transformation(img_gen)

            mse += self.mse_criterion(img_gen, img[t])
            kld += kl_criterion(mu, logvar, self.batch_size)

        loss = mse + self.kl_annealing.get_beta() * kld  # type: ignore
        loss.backward()  # type: ignore
        self.optimizer_step()

        return (
            loss / self.train_vi_len,
            mse / self.train_vi_len,
            kld / self.train_vi_len,
        )

    def val_one_step(self, img, label):
        img = img.permute(1, 0, 2, 3, 4)  # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4)  # change tensor into (seq, B, C, H, W)

        decoded_frame_list = [img[0].cpu()]
        label_list = []

        # Normal normal
        last_human_feat = self.frame_transformation(img[0])
        first_templete = last_human_feat.clone()
        out = img[0]

        for i in range(1, self.val_vi_len):
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()  # type: ignore
            label_feat = self.label_transformation(label[i])
            human_feat_hat = self.frame_transformation(out)

            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)
            out = self.Generator(parm)

            decoded_frame_list.append(out.cpu())
            label_list.append(label[i].cpu())

        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        label_frame = stack(label_list).permute(1, 0, 2, 3, 4)
        gen_image = generated_frame[0]

        PSNR_LIST = []
        for i in range(1, self.val_vi_len):
            # img[i] = img[i].squeeze(dim=0)
            # gen_image[i] = gen_image[i].squeeze(dim=0)
            PSNR = Generate_PSNR(img[i], gen_image[i])
            PSNR_LIST.append(PSNR.item())

        with open("./{}/frame_PSNR_record.txt".format(args.log_dir), "a") as f:
            for i in range(len(PSNR_LIST)):
                f.write(f"{PSNR_LIST[i]:.5f}\n")

        psnr = sum(PSNR_LIST) / (len(PSNR_LIST) - 1)

        return psnr

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))

        new_list[0].save(
            img_name,
            format="GIF",
            append_images=new_list,
            save_all=True,
            duration=40,
            loop=0,
        )

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor(),
            ]
        )

        dataset = Dataset_Dance(
            root=self.args.DR,
            transform=transform,
            mode="train",
            video_len=self.train_vi_len,
            partial=args.fast_partial if self.args.fast_train else args.partial,
        )
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            shuffle=False,
        )
        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor(),
            ]
        )
        dataset = Dataset_Dance(
            root=self.args.DR,
            transform=transform,
            mode="val",
            video_len=self.val_vi_len,
            partial=1.0,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.args.num_workers,
            drop_last=True,
            shuffle=False,
        )
        return val_loader

    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch > self.args.tfr_sde:
            self.tfr = max(self.tfr - self.tfr_d_step, 0)

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(
            f"({mode}) Epoch {self.current_epoch}, lr:{lr}", refresh=False
        )
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    def save(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "optimizer": self.state_dict(),
                "lr": self.scheduler.get_last_lr()[0],
                "tfr": self.tfr,
                "last_epoch": self.current_epoch,
            },
            path,
        )
        print(f"save ckpt to {path}, lr: {self.scheduler.get_last_lr()[0]}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint["state_dict"], strict=True)
            self.args.lr = 1e-5
            self.tfr = checkpoint["tfr"]
            self.current_epoch = checkpoint["last_epoch"]

            self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optim, step_size=3, gamma=0.1
            )
            self.kl_annealing = kl_annealing(
                self.args, current_epoch=checkpoint["last_epoch"]
            )

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()


def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    seed = 84
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--optim", type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--store_visualization",
        action="store_true",
        help="If you want to see the result while training",
    )
    parser.add_argument("--DR", type=str, required=True, help="Your Dataset Path")
    parser.add_argument(
        "--save_root", type=str, required=True, help="The path to save your data"
    )
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument(
        "--num_epoch", type=int, default=100, help="number of total epoch"
    )
    parser.add_argument(
        "--per_save", type=int, default=1, help="Save checkpoint every seted epoch"
    )
    parser.add_argument(
        "--partial",
        type=float,
        default=1.0,
        help="Part of the training dataset to be trained",
    )
    parser.add_argument(
        "--train_vi_len", type=int, default=16, help="Training video length"
    )
    parser.add_argument(
        "--val_vi_len", type=int, default=630, help="valdation video length"
    )
    parser.add_argument(
        "--frame_H", type=int, default=32, help="Height input image to be resize"
    )
    parser.add_argument(
        "--frame_W", type=int, default=64, help="Width input image to be resize"
    )

    # Module parameters setting
    parser.add_argument(
        "--F_dim", type=int, default=128, help="Dimension of feature human frame"
    )
    parser.add_argument(
        "--L_dim", type=int, default=32, help="Dimension of feature label frame"
    )
    parser.add_argument("--N_dim", type=int, default=12, help="Dimension of the Noise")
    parser.add_argument(
        "--D_out_dim",
        type=int,
        default=192,
        help="Dimension of the output in Decoder_Fusion",
    )

    # Teacher Forcing strategy
    parser.add_argument(
        "--tfr", type=float, default=1.0, help="The initial teacher forcing ratio"
    )
    parser.add_argument(
        "--tfr_sde",
        type=int,
        default=10,
        help="The epoch that teacher forcing ratio start to decay",
    )
    parser.add_argument(
        "--tfr_d_step",
        type=float,
        default=0.05,
        help="Decay step that teacher forcing ratio adopted",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="The path of your checkpoints"
    )

    # Training Strategy
    parser.add_argument("--fast_train", action="store_true")
    parser.add_argument(
        "--fast_partial",
        type=float,
        default=0.4,
        help="Use part of the training data to fasten the convergence",
    )
    parser.add_argument(
        "--fast_train_epoch",
        type=int,
        default=100,
        help="Number of epoch to use fast train mode",
    )

    # Kl annealing stratedy arguments
    parser.add_argument(
        "--kl_anneal_type", type=str, default="Monotonic", help="Cyclical or Monotonic"
    )
    parser.add_argument("--kl_anneal_cycle", type=int, default=10, help="")
    parser.add_argument("--kl_anneal_ratio", type=float, default=1, help="")
    parser.add_argument(
        "--beta", type=float, default=0.000, help="weighting on KL to prior"
    )
    parser.add_argument(
        "--log_dir", default="logs/fp", help="base directory to save logs"
    )

    args = parser.parse_args()

    main(args)
