from abc import ABC, abstractmethod

import torch
import numpy as np

from models.models import KnapsackExtractWeightsCostFromEmbeddingMLP, baseline_mlp_dict, KnapsackExtractWeightsFromFeatures
from models.modules import get_solver_module, StaticConstraintModule, CvxpyModule, CombOptNetModule
from utils.utils import loss_from_string, optimizer_from_string, set_seed, AvgMeters, compute_metrics, \
    knapsack_round, compute_normalized_solution, compute_denormalized_solution, solve_unconstrained


def get_trainer(trainer_name, **trainer_params):
    trainer_dict = dict(MLPTrainer=MLPBaselineTrainer,
                        KnapsackConstraintLearningTrainer=KnapsackConstraintLearningTrainer,
                        RandomConstraintLearningTrainer=RandomConstraintLearningTrainer,
                        KnapsackWeightsLearningTrainer=KnapsackWeightsLearningTrainer)
    return trainer_dict[trainer_name](**trainer_params)


class BaseTrainer(ABC):
    def __init__(self, train_iterator, test_iterator, use_cuda, optimizer_name, loss_name, optimizer_params, metadata,
                 model_params, seed, penalty=None):

        self._penalty = penalty
        set_seed(seed)
        self.use_cuda = use_cuda
        self.device = 'cuda' if self.use_cuda else 'cpu'

        self.train_iterator = train_iterator
        self.test_iterator = test_iterator

        self.true_variable_range = metadata['variable_range']
        self.num_variables = metadata['num_variables']
        self.variable_range = self.true_variable_range

        model_params['metadata'] = metadata

        model_parameters = self.build_model(**model_params)
        self.optimizer = optimizer_from_string(optimizer_name)(model_parameters, **optimizer_params)
        self.loss_fn = loss_from_string(loss_name)

    @abstractmethod
    def build_model(self, **model_params):
        pass

    @abstractmethod
    def calculate_loss_metrics(self, **data_params):
        pass

    def train_epoch(self):
        self.train = True
        metrics = AvgMeters()

        for i, data in enumerate(self.train_iterator):
            x, y_true_norm = [dat.to(self.device) for dat in data]
            loss, metric_dct = self.calculate_loss_metrics(x=x, y_true_norm=y_true_norm)
            metrics.update(metric_dct, n=x.size(0))

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

        results = metrics.get_averages(prefix='train_')
        return results

    def evaluate(self):
        self.train = False
        metrics = AvgMeters()

        for i, data in enumerate(self.test_iterator):
            x, y_true_norm = [dat.to(self.device) for dat in data]
            loss, metric_dct = self.calculate_loss_metrics(x=x, y_true_norm=y_true_norm)
            metrics.update(metric_dct, n=x.size(0))

        results = metrics.get_averages(prefix='eval_')
        return results


# FIXME: not tested
class MLPBaselineTrainer(BaseTrainer):
    def build_model(self, model_name, **model_params):
        self.model = baseline_mlp_dict[model_name](num_variables=self.num_variables, **model_params).to(
            self.device)
        return self.model.parameters()

    def calculate_loss_metrics(self, x, y_true_norm):
        y_norm = self.model(x=x)
        loss = self.loss_fn(y_norm.double(), y_true_norm)

        metrics = dict(loss=loss.item())
        y_denorm = compute_denormalized_solution(y_norm, **self.variable_range)
        y_denorm_rounded = torch.round(y_denorm)
        y_true_denorm = compute_denormalized_solution(y_true_norm, **self.true_variable_range)
        metrics.update(compute_metrics(y=y_denorm_rounded, y_true=y_true_denorm))
        return loss, metrics


# FIXME: not tested
class ConstraintLearningTrainerBase(BaseTrainer, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def calculate_loss_metrics(self, x, y_true_norm):
        y_denorm, y_denorm_roudned, solutions_denorm_dict, cost_vector = self.forward(x)
        y_norm = compute_normalized_solution(y_denorm, **self.variable_range)
        loss = self.loss_fn(y_norm.double(), y_true_norm)

        metrics = dict(loss=loss.item())
        y_uncon_denorm = solve_unconstrained(cost_vector=cost_vector, **self.variable_range)
        y_true_denorm = compute_denormalized_solution(y_true_norm, **self.true_variable_range)
        metrics.update(compute_metrics(y=y_denorm_roudned, y_true=y_true_denorm, y_uncon=y_uncon_denorm))
        for prefix, solution in solutions_denorm_dict.items():
            metrics.update(
                compute_metrics(y=solution, y_true=y_true_denorm, y_uncon=y_uncon_denorm, prefix=prefix + "_"))
        return loss, metrics


# FIXME: not tested
class RandomConstraintLearningTrainer(ConstraintLearningTrainerBase):
    def build_model(self, constraint_module_params, solver_module_params):
        self.static_constraint_module = StaticConstraintModule(variable_range=self.variable_range,
                                                               num_variables=self.num_variables,
                                                               **constraint_module_params).to(self.device)
        self.solver_module = get_solver_module(variable_range=self.variable_range,
                                               **solver_module_params).to(self.device)
        self.ilp_solver_module = CombOptNetModule(variable_range=self.variable_range).to(self.device)
        model_parameters = list(self.static_constraint_module.parameters()) + list(self.solver_module.parameters())
        return model_parameters

    def forward(self, x):
        cost_vector = x
        cost_vector = cost_vector / torch.norm(cost_vector, p=2, dim=-1, keepdim=True)
        constraints = self.static_constraint_module()

        y_denorm = self.solver_module(cost_vector=cost_vector, constraints=constraints)
        y_denorm_rounded = torch.round(y_denorm)
        solutions_dict = {}

        if not self.train and isinstance(self.solver_module, CvxpyModule):
            y_denorm_ilp = self.ilp_solver_module(cost_vector=cost_vector, constraints=constraints)
            update_dict = dict(ilp_postprocess=y_denorm_ilp)
            solutions_dict.update(update_dict)

        return y_denorm, y_denorm_rounded, solutions_dict, cost_vector


# FIXME: not tested
class KnapsackConstraintLearningTrainer(ConstraintLearningTrainerBase):
    def build_model(self, solver_module_params, backbone_module_params, metadata):
        self.backbone_module = KnapsackExtractWeightsCostFromEmbeddingMLP(**backbone_module_params).to(self.device)
        self.solver_module = get_solver_module(variable_range=self.variable_range,
                                               **solver_module_params).to(self.device)
        model_parameters = list(self.backbone_module.parameters()) + list(self.solver_module.parameters())
        return model_parameters

    def forward(self, x):
        cost_vector, constraints = self.backbone_module(x)
        cost_vector = cost_vector / torch.norm(cost_vector, p=2, dim=-1, keepdim=True)

        y_denorm = self.solver_module(cost_vector=cost_vector, constraints=constraints)
        if isinstance(self.solver_module, CvxpyModule):
            y_denorm_rounded = knapsack_round(y_denorm=y_denorm, constraints=constraints,
                                              knapsack_capacity=self.backbone_module.knapsack_capacity)
        else:
            y_denorm_rounded = y_denorm
        return y_denorm, y_denorm_rounded, {}, cost_vector


class KnapsackWeightsLearningTrainer(ConstraintLearningTrainerBase):
    def build_model(self, solver_module_params, backbone_module_params, metadata):

        self._in_features = backbone_module_params['in_features']
        self._kp_dim = backbone_module_params['out_features']

        self.backbone_module = \
            KnapsackExtractWeightsFromFeatures(kp_dim=self._kp_dim,
                                               embed_dim=backbone_module_params['in_features'],
                                               knapsack_capacity=metadata['capacity'],
                                               weight_min=metadata['min_weight'],
                                               weight_max=metadata['max_weight'],
                                               out_features=backbone_module_params['out_features']).to(self.device)
        self.solver_module = get_solver_module(variable_range=self.variable_range,
                                               **solver_module_params).to(self.device)
        model_parameters = list(self.backbone_module.parameters()) + list(self.solver_module.parameters())
        return model_parameters

    def forward(self, x):

        # FIXME: this should not be hardcoded
        assert x.shape[1] == self._in_features + (self._kp_dim * 2) + 1

        in_features = x[:, :self._in_features]
        cost_vector = x[:, self._in_features:self._in_features+self._kp_dim]
        weights = x[:, self._in_features+self._kp_dim:-1]
        capacity = x[:, -1]

        constraints = self.backbone_module(in_features)

        y_denorm = self.solver_module(cost_vector=-cost_vector, constraints=constraints)
        if isinstance(self.solver_module, CvxpyModule):
            y_denorm_rounded = knapsack_round(y_denorm=y_denorm, constraints=constraints,
                                              knapsack_capacity=self.backbone_module.knapsack_capacity)
        else:
            y_denorm_rounded = y_denorm
        return y_denorm, y_denorm_rounded, {}, cost_vector, weights, capacity

    def calculate_loss_metrics(self, x, y_true_norm):
        y_denorm, y_denorm_roudned, solutions_denorm_dict, cost_vector, weights, capacity = self.forward(x)
        y_norm = compute_normalized_solution(y_denorm, **self.variable_range)
        loss = self.loss_fn(y_norm.double(), y_true_norm)

        metrics = dict(loss=loss.item())
        y_uncon_denorm = solve_unconstrained(cost_vector=cost_vector, **self.variable_range)
        y_true_denorm = compute_denormalized_solution(y_true_norm, **self.true_variable_range)
        metrics.update(compute_metrics(y=y_denorm_roudned, y_true=y_true_denorm, y_uncon=y_uncon_denorm))
        for prefix, solution in solutions_denorm_dict.items():
            metrics.update(
                compute_metrics(y=solution, y_true=y_true_denorm, y_uncon=y_uncon_denorm, prefix=prefix + "_"))

        y = y_denorm.cpu().detach().numpy()
        y_true = y_true_denorm.cpu().detach().numpy()
        weights = weights.cpu().detach().numpy()
        values = cost_vector.cpu().detach().numpy()
        capacity = capacity.cpu().detach().numpy()

        rel_regret = list()
        subopt_rel_regret = list()
        infeas_ratio = 0

        for y_i, y_true_i, weights_i, values_i, capacity_i in zip(y, y_true, weights, values, capacity):

            pred_tot_weight = np.matmul(y_i, weights_i)

            # Correct the solution if needed
            corrected_y_i = y_i.copy()

            # Apply the correction action if the total weight exceeds the capacity
            if pred_tot_weight > capacity_i:

                # Sort items by decreasing value of weight
                sorted_weights_indexes = np.argsort(-weights_i)

                # The correction action consists of discarding one element at a time by decreasing order of weight until the
                # capacity is not overcome
                for idx in sorted_weights_indexes:

                    # Discard the item
                    corrected_y_i[idx] = 0

                    # If capacity is not overcome then stop
                    if np.matmul(corrected_y_i, weights_i) <= capacity_i:
                        break

            # Check which items have been discarded
            discarded_items = y_i - corrected_y_i
            # Pay additional cost for discarded items
            discarded_items_cost = -np.matmul(discarded_items, self._penalty * values_i)

            # Compute the cost of the corrected solution
            suboptimality_cost = np.matmul(corrected_y_i, values_i)

            cost_i = discarded_items_cost + suboptimality_cost
            true_cost_i = np.matmul(y_true_i, values_i)
            rel_regret_i = (true_cost_i - cost_i) / true_cost_i
            subopt_rel_regret_i = (true_cost_i - suboptimality_cost) / true_cost_i
            subopt_rel_regret.append(subopt_rel_regret_i)
            rel_regret.append(rel_regret_i)

            if discarded_items_cost != 0:
                infeas_ratio += 1

        avg_rel_regret = np.mean(rel_regret)
        metrics['rel_regret'] = avg_rel_regret
        avg_subopt_rel_regret = np.mean(subopt_rel_regret)
        metrics['subopt_rel_regret'] = avg_subopt_rel_regret
        infeas_ratio = infeas_ratio / len(y)
        metrics['infeas_ratio'] = infeas_ratio

        return loss, metrics
