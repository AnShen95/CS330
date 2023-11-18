"""Implementation of model-agnostic meta-learning for Omniglot."""
import sys
sys.path.append('..')
import argparse
import os

import numpy as np
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard
from google_drive_downloader import GoogleDriveDownloader as gdd

import load_data

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 32
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 600




def eval(y_pred, Y):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(Y)):
        if Y[i][0] == 1:
            if round(y_pred[i][0]) == 1:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if round(y_pred[i][0]) == 1:
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


class MAML:
    """Trains and assesses a MAML."""

    def __init__(
            self,
            NUM_LAYERS,
            NUM_NEURON,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            log_dir,
            device
    ):
        """Inits MAML.

        The network consists of four convolutional blocks followed by a linear
        head layer. Each convolutional block comprises a convolution layer, a
        batch normalization layer, and ReLU activation.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.

        Args:
            num_outputs (int): dimensionality of output, i.e. number of classes
                in a task
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
                If learn_inner_lrs=True, inner_lr serves as the initialization
                of the learning rates.
            learn_inner_lrs (bool): whether to learn the above
            outer_lr (float): learning rate for outer-loop optimization
            log_dir (str): path to logging directory
            device (str): device to be used
        """
        meta_parameters = {}

        self.device = device

        meta_parameters[f'w0'] = nn.init.xavier_uniform_(
            torch.empty(
                NUM_NEURON,
                18,
                requires_grad=True,
                device=self.device
            )
        )
        meta_parameters[f'b0'] = nn.init.zeros_(
            torch.empty(
                NUM_NEURON,
                requires_grad=True,
                device=self.device
            )
        )

        for i in range(NUM_LAYERS - 2):
            meta_parameters[f'w{i + 1}'] = nn.init.xavier_uniform_(
                torch.empty(
                    NUM_NEURON,
                    NUM_NEURON,
                    requires_grad=True,
                    device=self.device
                )
            )
            meta_parameters[f'b{i + 1}'] = nn.init.zeros_(
                torch.empty(
                    NUM_NEURON,
                    requires_grad=True,
                    device=self.device
                )
            )

        meta_parameters[f'w{NUM_LAYERS - 1}'] = nn.init.xavier_uniform_(
            torch.empty(
                1,
                NUM_NEURON,
                requires_grad=True,
                device=self.device
            )
        )
        meta_parameters[f'b{NUM_LAYERS - 1}'] = nn.init.zeros_(
            torch.empty(
                1,
                requires_grad=True,
                device=self.device
            )
        )
        self._num_layer = NUM_LAYERS
        self._meta_parameters = meta_parameters
        self._num_inner_steps = num_inner_steps
        self._inner_lrs = {
            k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs)
            for k in self._meta_parameters.keys()
        }
        self._outer_lr = outer_lr

        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _forward(self, input, parameters):
        """Computes predicted classification logits.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)
            parameters (dict[str, Tensor]): parameters to use for
                the computation

        Returns:
            a Tensor consisting of a batch of logits
                shape (num_images, classes)
        """
        x =  F.linear(input=input,weight=parameters[f'w0'],bias=parameters[f'b0'])
        x = F.relu(x)
        for i in range(self._num_layer - 2):
            x = F.linear(input=x, weight=parameters[f'w{i + 1}'], bias=parameters[f'b{i + 1}'])
            x = F.relu(x)
        x = F.linear(input=x, weight=parameters[f'w{self._num_layer - 1}'], bias=parameters[f'b{self._num_layer - 1}'])
        return x

    def _inner_loop(self, images, labels, train):
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            images (Tensor): task support set inputs
                shape (num_images, channels, height, width)
            labels (Tensor): task support set outputs
                shape (num_images,)
            train (bool): whether we are training or evaluating

        Returns:
            parameters (dict[str, Tensor]): adapted network parameters
            accuracies (list[float]): support set accuracy over the course of
                the inner loop, length num_inner_steps + 1
            gradients(list[float]): gradients computed from auto.grad, just needed
                for autograders, no need to use this value in your code and feel to replace
                with underscore       
        """
        accuracies = []
        parameters = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }
        gradients = None
        ### START CODE HERE ###
        # TODO: finish implementing this method.
        # This method computes the inner loop (adaptation) procedure
        # over the course of _num_inner_steps steps for one
        # task. It also scores the model along the way.
        # Make sure to populate accuracies and update parameters.
        # Use F.cross_entropy to compute classification losses.
        # Use util.score to compute accuracies.
        gradients=[]
        for epoch in range(self._num_inner_steps):
            preds = self._forward(images, parameters)
            #accuracies.append(util.score(preds, labels))
            accuracies.append(0)
            loss = F.binary_cross_entropy_with_logits(preds, labels)
            for k, v in parameters.items():
                g = torch.autograd.grad(loss, v, retain_graph=True,  create_graph=train)
                gradients.append(g[0])
                parameters[k] = v - g[0] * self._inner_lrs[k]
                preds = self._forward(images, parameters)
            accuracies.append(0)

        ### END CODE HERE ###
        return parameters, accuracies, gradients

    def _outer_step(self, task_batch, train):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from an Omniglot DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch, scalar
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """
        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []

        images_support, labels_support, images_query, labels_query = task_batch
        images_support = images_support.to(self.device)
        labels_support = labels_support.to(self.device)
        images_query = images_query.to(self.device)
        labels_query = labels_query.to(self.device)
        ### START CODE HERE ###
        # TODO: finish implementing this method.
        # For a given task, use the _inner_loop method to adapt for
        # _num_inner_steps steps, then compute the MAML loss and other
        # metrics. Reminder you can replace gradients with _ when calling
        # _inner_loop.
        # Use F.cross_entropy to compute classification losses.
        # Use util.score to compute accuracies.
        # Make sure to populate outer_loss_batch, accuracies_support_batch,
        # and accuracy_query_batch.
        # support accuracy: The first element (index 0) should be the accuracy before any steps are taken.

        parameters, accuracies_support, _ = self._inner_loop(images_support, labels_support, train)
        accuracies_support_batch.append(accuracies_support)

        preds = self._forward(images_query, parameters)
        preds = F.sigmoid(preds)
        l = F.binary_cross_entropy(preds, labels_query)

        return l, preds, labels_query

    def train(self, dataloader_meta_train, dataloader_meta_val, writer):
        """Train the MAML.

        Consumes dataloader_meta_train to optimize MAML meta-parameters
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
            outer_loss, _,_ = self._outer_step(task_batch, train=True)
            outer_loss.backward()
            self._optimizer.step()

            if i_step % VAL_INTERVAL == 0:
                losses = []
                preds = []
                labels = []
                for _ in range(100):
                    val_task_batch = dataloader_meta_val._sample()
                    outer_loss, pred, label = (
                        self._outer_step(val_task_batch, train=False)
                    )
                    losses.append(outer_loss.item())
                    preds.extend(pred.tolist())
                    labels.extend(label.tolist())
                loss = np.mean(losses)
                F1, p, r, accu = eval(preds, labels)
                if F1 > maxF1:
                    maxF1 = F1

                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar("F1/test", F1, i_step)
        print("Max F1: " + str(maxF1))

    def test(self, dataloader_test):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            _, _, accuracy_query = self._outer_step(task_batch, train=False)
            accuracies.append(accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._meta_parameters = state['meta_parameters']
            self._inner_lrs = state['inner_lrs']
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        optimizer_state_dict = self._optimizer.state_dict()
        torch.save(
            dict(meta_parameters=self._meta_parameters,
                 inner_lrs=self._inner_lrs,
                 optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):

    print(args)

    if args.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # on MPS the derivative for aten::linear_backward is not implemented ... Waiting for PyTorch 2.1.0
        # DEVICE = "mps"

        # Due to the above, default for now to cpu
        DEVICE = "cpu"
    elif args.device == "gpu" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print("Using device: ", DEVICE)

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/maml/S{args.num_support}.Q{args.num_query}.L{args.num_layer}.N{args.num_neuron}.B{args.batch_size}.US.{args.data_type}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    maml = MAML(
        args.num_layer,
        args.num_neuron,
        args.num_inner_steps,
        args.inner_lr,
        args.learn_inner_lrs,
        args.outer_lr,
        log_dir,
        DEVICE
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on {num_training_tasks} tasks with composition: '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )

        dataloader_meta_train = load_data.DataGenerator(
            args.num_way,
            args.num_query + 1,
            batch_type="train",
            file_name=f"C:\\Temp\\CS330\\Project\\data\\american_bankruptcy_{args.data_type}.csv",
            DEVICE=DEVICE,
            has_title=True,
        )
        dataloader_meta_val = load_data.DataGenerator(
            args.num_way,
            args.num_query + 1,
            batch_type="test",
            file_name=r"C:\Temp\CS330\Project\data\american_bankruptcy_normal.csv",
            DEVICE=DEVICE,
            has_title=True,
        )
        maml.train(
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
        dataloader_test = load_data.DataGenerator(
            args.num_way,
            args.num_query + 1,
            batch_type="test",
            file_name=r"C:\Temp\CS330\Project\data\american_bankruptcy_normal.csv",
            DEVICE=DEVICE,
            has_title=True,
        )
        maml.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument("--data_type", type=str, default="normal")
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=1,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=20,
                        help='number of query examples per class in a task')
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--num_neuron", type=int, default=64)
    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.4,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                        help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='outer-loop learning rate')
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
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--device', type=str, default='gpu')

    args = parser.parse_args()

    main(args)