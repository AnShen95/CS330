"""Implementation of prototypical networks for Omniglot."""
import sys
sys.path.append('..')
import argparse
import os

import numpy as np
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch import nn
import torch.nn.functional as F  # pylint: disable=unused-import
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils import tensorboard

import load_data

NUM_INPUT_CHANNELS = 1
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5
NUM_TEST_TASKS = 600

def eval(y_pred, Y):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(Y)):
        if Y[i][0] == 0:
            if round(y_pred[i][0].item()) == 1:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if round(y_pred[i][0].item()) == 1:
                FP = FP + 1
            else:
                TN = TN + 1
    try:
        p = TP/(TP + FP)
        r = TP/(TP + FN)
        F1 = 2 * p * r / (p + r)
        accu = (TP+TN) / (TP+TN+FP+FN)

        return F1,p,r,accu
    except:
        return 0,0,0,0


class ProtoNetNetwork(nn.Module):
    """Container for ProtoNet weights and image-to-latent computation."""

    def __init__(self,
                 NUM_LAYERS,
                 NUM_NEURON,
                 NUM_DIMENSION,
                 device):
        """Inits ProtoNetNetwork.

        The network consists of four convolutional blocks, each comprising a
        convolution layer, a batch normalization layer, ReLU activation, and 2x2
        max pooling for downsampling. There is an additional flattening
        operation at the end.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.

        Args:
            device (str): device to be used
        """
        super().__init__()
        layers = []
        layers.append(nn.Linear(18, NUM_NEURON))
        layers.append(nn.ReLU())
        for _ in range(NUM_LAYERS - 2):
            layers.append(nn.Linear(NUM_NEURON, NUM_NEURON))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(NUM_NEURON, NUM_DIMENSION))
        self._layers = nn.Sequential(*layers)
        self.to(device)

    def forward(self, images):
        """Computes the latent representation of a batch of images.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)

        Returns:
            a Tensor containing a batch of latent representations
                shape (num_images, latents)
        """
        return self._layers(images)


class ProtoNet:
    """Trains and assesses a prototypical network."""

    def __init__(self, learning_rate, log_dir, device, compile=False, backend=None, learner=None, val_interval=None, save_interval=None, bio=False):
        """Inits ProtoNet.

        Args:
            learning_rate (float): learning rate for the Adam optimizer
            log_dir (str): path to logging directory
            device (str): device to be used
        """
        self.device = device
        if learner is None:
            self._network = ProtoNetNetwork(args.num_layer, args.num_neuron, args.num_dimension, device)
        else: 
            self._network = learner.to(device)

        self.val_interval = VAL_INTERVAL if val_interval is None else val_interval
        self.save_interval = SAVE_INTERVAL if save_interval is None else save_interval
        self.bio = bio

        if(compile == True):
            try:
                self._network = torch.compile(self._network, backend=backend)
                print(f"ProtoNetNetwork model compiled")
            except Exception as err:
                print(f"Model compile not supported: {err}")

        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _step(self, task_batch):
        """Computes ProtoNet mean loss (and accuracy) on a batch of tasks.

        Args:
            task_batch (tuple[Tensor, Tensor, Tensor, Tensor]):
                batch of tasks from an Omniglot DataLoader

        Returns:
            a Tensor containing mean ProtoNet loss over the batch
                shape ()
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """
        images_support, labels_support, images_query, labels_query = task_batch
        images_support = images_support.to(self.device)
        labels_support = labels_support.to(self.device)
        images_query = images_query.to(self.device)
        labels_query = labels_query.to(self.device)

        feature_support = self._network(images_support)

        ptototype_raw = {}
        for i in range(len(labels_support)):
            label = labels_support[i].item()
            if label not in ptototype_raw:
                ptototype_raw[label] = []
            ptototype_raw[label].append(feature_support[i])
        ptototype_stacked = {}
        for key,value in ptototype_raw.items():
            ptototype_stacked[key] = torch.stack(ptototype_raw[key])
        ptototype_dict = {}
        for key,value in ptototype_stacked.items():
            ptototype_dict[key] = torch.mean(ptototype_stacked[key],dim=0)
        ptototype_list = []
        for i in range(len(ptototype_dict)):
            ptototype_list.append(ptototype_dict[i])
        ptototype = torch.stack(ptototype_list)

        feature_query = self._network(images_query)
        pred_query_list = []
        loss_query_list = []
        for i in range(len(labels_query)):
            distance_query = feature_query[i] - ptototype
            distance_query = torch.pow(distance_query, 2)
            distance_query = torch.sum(distance_query, dim=-1)
            preds = F.softmax(-distance_query)
            loss = F.cross_entropy(preds, labels_query[i][0])
            loss_query_list.append(loss)
            pred_query_list.append(preds)
            ### END CODE HERE ###
        return (torch.mean(torch.stack(loss_query_list)), pred_query_list,labels_query)

    def train(self, dataloader_meta_train, dataloader_meta_val, writer):
        """Train the ProtoNet.

        Consumes dataloader_meta_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_meta_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_meta_train (DataLoader): loader for train tasks
            dataloader_meta_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        maxF1 = 0
        for i_step in range(args.num_train_iterations):
            task_batch = dataloader_meta_train._sample()
            self._optimizer.zero_grad()
            loss, _, _ = self._step(task_batch)
            loss.backward()
            self._optimizer.step()

            if i_step % self.val_interval == 0:
                with torch.no_grad():
                    losses = []
                    preds = []
                    labels = []
                    for _ in range(100):
                        val_task_batch = dataloader_meta_val._sample()
                        loss, pred, label = self._step(val_task_batch)
                        losses.append(loss.item())
                        preds.extend(pred)
                        labels.extend(label.tolist())
                    loss = np.mean(losses)
                    F1, p, r, accu = eval(preds, labels)
                    if F1 > maxF1:
                        maxF1 = F1

                    writer.add_scalar('loss/val', loss, i_step)
                    writer.add_scalar("F1/test", F1, i_step)
        print("Max F1: " + str(maxF1))


    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for i, task_batch in enumerate(dataloader_test):
            accuracies.append(self._step(task_batch)[2])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step, filename=""):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load
            filename (str): directly setting name of checkpoint file, default ="", when argument is passed, then checkpoint will be ignored

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        ) if filename == "" else filename
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._network.load_state_dict(state['network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):

    print(args)

    if args.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = "mps"
    elif args.device == "gpu" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print("Using device: ", DEVICE)

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/protonet/S{args.num_support}.Q{args.num_query}.L{args.num_layer}.N{args.num_neuron}.D{args.num_dimension}.B{args.batch_size}.US.{args.data_type}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    protonet = ProtoNet(args.learning_rate, log_dir, DEVICE, args.compile, args.backend)

    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_meta_train = load_data.ProtoDataGenerator(
            args.num_way,
            args.num_query + 1,
            batch_type="train",
            file_name=f"C:\\Temp\\CS330\\Project\\data\\american_bankruptcy_{args.data_type}.csv",
            DEVICE=DEVICE,
            has_title=True,
        )
        dataloader_meta_val = load_data.ProtoDataGenerator(
            args.num_way,
            args.num_query + 1,
            batch_type="test",
            file_name=r"C:\Temp\CS330\Project\data\american_bankruptcy_normal.csv",
            DEVICE=DEVICE,
            has_title=True,
        )
        protonet.train(
            dataloader_meta_train,
            dataloader_meta_val,
            writer
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = load_data.ProtoDataGenerator(
            args.num_way,
            args.num_query + 1,
            batch_type="test",
            file_name=r"C:\Temp\CS330\Project\data\american_bankruptcy_normal.csv",
            DEVICE=DEVICE,
            has_title=True,
        )
        protonet.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument("--data_type", type=str, default="normal")
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=2,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=2,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=20,
                        help='number of query examples per class in a task')
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--num_neuron", type=int, default=64)
    parser.add_argument("--num_dimension", type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=100000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--num_workers', type=int, default=2, 
                        help=('needed to specify omniglot dataloader'))
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument("--backend", type=str, default="inductor", choices=['inductor', 'aot_eager', 'cudagraphs'])
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--device', type=str, default='gpu')

    args = parser.parse_args()

    main(args)

