from collections import OrderedDict

import torch
from torch import nn

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


class LiverMAML:

    def __init__(
        self,
        model: nn.Module,
        task_distribution,
        device: torch.device,
        alpha: float = 1e-3,
        beta: float = 1e-4,
        k_support: int = 4096,
        k_query: int = 4096,
        num_metatasks: int = 4,
        inner_steps: int = 1,
        first_order: bool = False,
    ):
        self.model = model.to(device)
        self.task_distribution = task_distribution
        self.device = device

        self.alpha = alpha
        self.beta = beta
        self.k_support = k_support
        self.k_query = k_query
        self.num_metatasks = num_metatasks
        self.inner_steps = inner_steps
        self.first_order = first_order

        self.criterion = nn.BCEWithLogitsLoss()
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=beta)
        self.meta_losses = []

    def named_parameters_dict(self):
        return OrderedDict(
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        )

    def forward_with_params(self, x, params):
        return functional_call(self.model, params, (x,))

    def inner_loop(self, task):
        params = self.named_parameters_dict()

        for _ in range(self.inner_steps):
            x_support, y_support = task.sample_data(self.k_support, self.device)

            support_logits = self.forward_with_params(x_support, params)
            support_loss = self.criterion(support_logits, y_support)

            grads = torch.autograd.grad(
                support_loss,
                params.values(),
                create_graph=not self.first_order,
                retain_graph=not self.first_order,
            )

            params = OrderedDict(
                (name, param - self.alpha * grad)
                for (name, param), grad in zip(params.items(), grads)
            )

        x_query, y_query = task.sample_data(self.k_query, self.device)
        query_logits = self.forward_with_params(x_query, params)
        query_loss = self.criterion(query_logits, y_query)

        return query_loss

    # 여기 (아마도) meta-learning loop; 서버 사양 봐가면서 조정해보기
    def outer_loop(self, num_iterations: int, print_every: int = 10):
        self.model.train()

        for iteration in range(1, num_iterations + 1):
            self.meta_optimizer.zero_grad(set_to_none=True)

            meta_loss = 0.0
            # 1: Need large num_iterations to cover all the cases of the training set
            for _ in range(self.num_metatasks):
                task = self.task_distribution.sample_task()
                task_loss = self.inner_loop(task)
                meta_loss = meta_loss + task_loss

            meta_loss = meta_loss / self.num_metatasks
            meta_loss.backward()
            
            # 2 Cover all the training set but need to tune to num_metatasks to adapt/fit with the GPU VRAM
            for batch in trainingset # fix it; 
                for _ in range(self.num_metatasks):
                    task = self.task_distribution.sample_task()
                    task_loss = self.inner_loop(task)
                    meta_loss = meta_loss + task_loss

                meta_loss = meta_loss / self.num_metatasks
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.meta_optimizer.step()

                loss_value = float(meta_loss.detach().cpu())
                self.meta_losses.append(loss_value)

            if iteration % print_every == 0:
                print(f"[{iteration}/{num_iterations}] meta_loss={loss_value:.6f}")

        return self.meta_losses