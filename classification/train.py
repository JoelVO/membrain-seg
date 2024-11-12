from membrain_seg.segmentation.networks.unet import SemanticSegmentationUnet
from loader import Loader
import torch
from tqdm import trange, tqdm
import sys
import numpy as np


class TrainingClass:
    def __init__(self, data_path: str, crop_size: tuple, out_channels: int = 16,
                 device: torch.device = torch.device('cpu'),
                 memory_bank_size: int = 32):
        """
        Parameters
        ----------
        data_path: path where to find the data structure.
        crop_size: desired size for training crops.
        out_channels: token's size for encoded data.
        device: device where to send model and data to.
        memory_bank_size: size for the memory bank when training.
        """
        self._model = SemanticSegmentationUnet(use_deep_supervision=False,
                                               out_channels=out_channels,
                                               learning_rate=1e-3)
        self.loader = Loader(data_path, crop_size)
        optimizer, scheduler = self._model.configure_optimizers()
        self.optimizer = optimizer[0]
        self.scheduler = scheduler[0]
        self.device = device
        self.memory_bank_size = memory_bank_size

        memory_bank = self.loader(0, make_memory_bank=True,
                                  memory_bank_size=self.memory_bank_size)
        self.make_memory_bank(memory_bank)

    def __call__(self, x):
        return self._model(x)

    def crossentropy_loss(self, x: torch.tensor, y: torch.tensor):
        """
        Parameters
        ----------
        x: encoded crop.
        y: encoded crop.

        Returns
        -------
        Mean cross entropy between both entries.
        """
        x = torch.nn.functional.softmax(x.flatten(1), dim=1)
        y = torch.nn.functional.log_softmax(y.flatten(1), dim=1)
        loss = torch.sum(x * y, dim=1)

        return -loss.mean()

    def koleo(self, embedding: torch.tensor, memory_bank: torch.tensor, eps=1e-8):
        """
        Parameters
        ----------
        embedding: embeddings
        memory_bank: current memory bank
        eps: epsilon to use in the normalization

        Returns
        -------
        Loss value penalizing concentration of embeddings. This is to promote a uniform distribution in the latent
        space.
        """
        embedding, memory_bank = embedding.flatten(1), memory_bank.flatten(1)
        student_output = torch.nn.functional.normalize(embedding,
                                                       eps=eps, p=2, dim=-1)
        memory_bank = torch.nn.functional.normalize(memory_bank,
                                                    eps=eps, p=2, dim=-1)

        dists = torch.cdist(student_output, memory_bank)
        loss = -torch.log(torch.amin(dists, dim=-1) + eps).mean()
        return loss

    def make_memory_bank(self, memory_bank: torch.tensor):
        """
        Parameters
        ----------
        memory_bank: current memory bank.

        Returns
        -------
        None
        This creates a new memory bank and stores it as an attribute of the class.
        """
        self.memory_bank = []
        with torch.no_grad():
            for batch in tqdm(memory_bank):
                self.memory_bank.append(self(batch.unsqueeze(0)))

        self.memory_bank = torch.cat(self.memory_bank)

    def training_step(self, steps_per_epoch: int, batch_size: int, epoch: int):
        self._model.train()
        progress_bar = trange(steps_per_epoch, desc='Beginning epoch', leave=True)
        mean_train_loss = torch.tensor(0.)
        for step in progress_bar:
            batch = self.loader(batch_size, device=self.device,
                                memory_bank_size=self.memory_bank_size)

            anchor, positive, negative = batch
            anchor_f, positive_f, negative_f = self(anchor), self(positive), self(negative)

            for n in negative_f:
                _ = np.random.choice(len(self.memory_bank))
                self.memory_bank[_] = n.detach().clone()

            anchor_f, positive_f = anchor_f.flatten(1), positive_f.flatten(1)
            loss = self.crossentropy_loss(anchor_f, positive_f) + 0.1 * self.koleo(anchor_f, self.memory_bank)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 3)
            optimizer = self.optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_train_loss += loss.detach().item()

            progress_bar.set_description(
                f'Epoch: {epoch} Error: {mean_train_loss.item() / (step + 1)}, Best: {self.best}')
            progress_bar.refresh()

        return mean_train_loss

    def do_train(self, save_path: str, epochs: int, steps_per_epoch: int, batch_size: int, patience: int):
        """
        Parameters
        ----------
        save_path: path where to save the model's weights.
        epochs: number of epochs to do.
        steps_per_epoch: number of steps in each epoch.
        batch_size: number of samples per batch.
        patience: amount of epochs to allow without loss value decreasing before stopping the training process.

        Returns
        -------
        None

        """
        self.best = torch.inf
        scheduler = self.scheduler
        early_stop = 0
        for epoch in range(epochs):
            train_loss = self.training_step(steps_per_epoch, batch_size, epoch)

            scheduler.step(train_loss)

            if self.best > train_loss:
                self.best = train_loss
                early_stop = 0
                torch.save(self._model.state_dict(), f'{save_path}/best.pt')
            else:
                early_stop += 1

            torch.save(self._model.state_dict(), f'{save_path}/last.pt')

            if early_stop == patience:
                break

            memory_bank = self.loader(0, make_memory_bank=True,
                                      memory_bank_size=self.memory_bank_size)

            # Update the memory bank to have a better representation of the encoding
            self.make_memory_bank(memory_bank)


if __name__ == "__main__":

    if torch.cuda.is_available():
        device_ = torch.device('cuda')
    else:
        device_ = torch.device('cpu')

    training_model = TrainingClass(data_path=sys.argv[1], crop_size=(64, 64, 64), device=device_)
    training_model.do_train(save_path=sys.argv[2],
                            epochs=int(sys.argv[3]),
                            steps_per_epoch=int(sys.argv[4]),
                            batch_size=int(sys.argv[5]),
                            patience=int(sys.argv[6]))
