import pytorch_lightning as L
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models


class SwinB(L.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.swin_b(**model_kwargs)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15], gamma=0.1, )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        #         imgs, labels = batch
        preds = self.model(batch['image'])
        loss = F.l1_loss(preds, batch['boneage'])

        self.log("%s_loss" % mode, loss, prog_bar=True)
        self.log("%s_acc" % mode, loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
