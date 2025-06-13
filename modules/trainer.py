import torch
from pathlib import Path
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data

from .utils import cycle
from torch.optim.lr_scheduler import LambdaLR
import image_comp.datasetDistribute0318 as datasetDistribute0318
import config
from pytorch_msssim import ms_ssim
import numpy as np
import time

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# trainer class
class Trainer(object):
    def __init__(
        self,
        rank,
        sample_steps,
        diffusion_model,
        train_loader,
        scheduler_function,
        ema_decay=0.995,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=10000,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=10,
        results_folder="./results",
        tensorboard_dir="./tensorboard_logs/diffusion-video/",
        model_name="model",
        val_num_of_batch=1,
        optimizer="adam",
        sample_mode="ddpm",
        lagrangian = 1,
    ):
        super().__init__()
        self.model = diffusion_model
        self.sample_mode = sample_mode
        self.sample_steps = sample_steps
        self.save_and_sample_every = save_and_sample_every

        self.train_num_steps = train_num_steps

        self.train_loader = train_loader
        self.lagrangian = lagrangian
        if optimizer == "adam":
            self.opt = Adam(self.model.parameters(), lr=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(self.model.parameters(), lr=train_lr)
        self.scheduler = LambdaLR(self.opt, lr_lambda=scheduler_function)

        self.step = 0
        self.device = rank
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.model_name = model_name

    def save(self):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            # "ema": self.ema_model.module.state_dict(),
        }
        idx = (self.step // self.save_and_sample_every) % 3
        torch.save(data, str(self.results_folder / f"{self.model_name}_{idx}.pt"))

    def load(self, idx=0, load_step=True):
        print("ffffffffffffffffffffffffffffffffffffffffffffffffff")
        data = torch.load(
            str("/root/yezhuang/diffusion_train/new_diffusion_train/weight/big-l1-vimeo-d64-t20000-b0.2-vbrFalse-noise-linear-aux0.9lpips/big-l1-vimeo-d64-t20000-b0.2-vbrFalse-noise-linear-aux0.9lpips_2.pt"),
            map_location=lambda storage, loc: storage,
        )
        if load_step:
            self.step = data["step"]
        try:
            self.model.module.load_state_dict(data["model"], strict=False)
        except:
            self.model.load_state_dict(data["model"], strict=False)
        # self.ema_model.module.load_state_dict(data["ema"], strict=False)

    def train(self):
        def load_img(img_path,batch_size1):
            train_set = datasetDistribute0318.ImageFolder(is_train=True, root=img_path)
            train_loader = data.DataLoader(
                dataset=train_set, batch_size=batch_size1, shuffle=True, num_workers=1)
            print('total images: {}; total batches: {}'.format(
                len(train_set), len(train_loader)))
            return train_loader
        train_loader = load_img(config.data_config["data_path"], config.batch_size)
        global data
        mse = torch.nn.MSELoss(reduction='mean')
        mse = mse.cuda()
        print("startttttttttttttttttttt")
        for self.step in range(self.train_num_steps):
            print("epoch",self.step)
            self.model.train()

            if (self.step >= self.scheduler_checkpoint_step) and (self.step != 0):
                self.scheduler.step()

            for batch, (imgAll, filename, filenamePre, filenameNext) in enumerate(train_loader):
                #bp_t0 = time.time()
                #if(batch % 50 == 0):
                #    bp50time_t1 = time.time-bp50time_
                #    print("[TRAIN] EPOCH [{}] batch:({}/{})".format(self.step+1, batch+1, len(train_loader)))
                
                self.opt.zero_grad()
                imgAll = torch.cat((imgAll[0], imgAll[1]), dim=0)
                imgPre = imgAll[:, 0:3, :, :]
                imgMid = imgAll[:, 3:6, :, :]
                imgNext = imgAll[:, 6:9, :, :]
                img = imgMid
                dataSide = (imgPre + imgNext ) / 2
                patches = img.cuda()
                dataSide = dataSide.cuda()

                #print("****************************")
                #print(img.shape)
                #print(dataSide.shape)
                
                loss_x,aloss_x = self.model(patches * 2.0 - 1.0, dataSide * 2.0 - 1.0)
                loss =  loss_x
                aloss = aloss_x
                #print("loss = ", loss)
                #print("aloss = ", aloss)
                loss.backward()
                aloss.backward()
                self.opt.step()
                #bp_t1 = time.time() - bp_t0
                if(batch % 10 == 0):
                #print("time: ", bp_t1)
                #bp50time_t1 = time.time-bp50time_
                    print("[TRAIN] EPOCH [{}] batch:({}/{}); loss:{:.6f}; aloss:{:.4f}".format(self.step+1, batch+1, len(train_loader), loss, aloss))



            if (self.step % self.save_and_sample_every == 0):
                test_set = datasetDistribute0318.ImageFolder(is_train=False, root="/root/yezhuang/work/192_foreman")
                # 1210 Load test
                kframe_loader = data.DataLoader(
                    dataset=test_set, batch_size=1, shuffle=False, num_workers=0)
                # kframe_loader = load_img("/root/yezhuang/work/174", 1)
                val_loss = []
                val_mse = []
                val_bpp = []
                val_msssim = []
                val_distortion = []

                for batch, (imgAll, filename, filenamePre, filenameNext) in enumerate(kframe_loader):
                    imgPre = imgAll[:, 0:3, :, :]
                    imgMid = imgAll[:, 3:6, :, :]
                    imgNext = imgAll[:, 6:9, :, :]

                    data1 = imgMid
                    dataSide = (imgPre + imgNext) / 2
                    data1 = data1.cuda()
                    dataSide = dataSide.cuda()
                    print("batch", batch)
                    #print("data1: ", data1.shape)
                    #print("dataSide: ", dataSide.shape)
                    compressed_x,  bpp , joint_bpp  = self.model.compress(
                        data1 * 2.0 - 1.0,
                        dataSide * 2.0 - 1.0,
                        sample_steps=200,
                        sample_mode="ddim",
                        bpp_return_mean=False,
                        init=torch.randn_like(data1) * 0.8
                    )
                    print("bpp:", bpp)
                    compressed_x = compressed_x.clamp(-1, 1) / 2.0 + 0.5

                    mse_dist = mse(data1, compressed_x)

                    msssim = ms_ssim(data1.clone().cpu(), compressed_x.clone().cpu(), data_range=1.0, size_average=True,win_size=9)
                    msssim_db = msssim

                    distortion = (1 - ms_ssim(data1.cpu(), compressed_x.cpu(), data_range=1.0, size_average=True,
                                              win_size=9))

                    loss = self.lagrangian * distortion * (
                                255 ** 2) + bpp  # multiplied by (255 ** 2) for distortion scaling

                    val_mse.append(mse_dist.item())

                    # val_bpp.append(bpp.item())
                    val_bpp.append(torch.mean(bpp).item())

                    # val_loss.append(loss.item())
                    val_loss.append(torch.mean(loss).item())

                    # val_msssim.append(msssim_db.item())
                    val_msssim.append(torch.mean(msssim_db).item())

                    # val_distortion.append(distortion.item())
                    val_distortion.append(torch.mean(distortion).item())

                val_loss_to_track = sum(val_loss) / len(val_loss)
                tracking = ['Epoch {}:'.format(self.step + 1),
                            'Loss = {:.4f},'.format(val_loss_to_track),
                            'BPP = {:.4f},'.format(sum(val_bpp) / len(val_bpp)),
                            'Distortion = {:.4f},'.format(sum(val_distortion) / len(val_distortion)),
                            'PSNR = {:.4f},'.format(10 * np.log10(1 / (sum(val_mse) / (len(val_mse))))),
                            'MS-SSIM = {:.4f}'.format(sum(val_msssim) / len(val_msssim))]
                print(" ".join(tracking))
                self.save()
        self.save()
        print("training completed")
