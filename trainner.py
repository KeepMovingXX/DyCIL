import time
import torch
import numpy as np
from tqdm import tqdm
import random
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

import warnings
warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Trainer(object):
    def __init__(self, args, model, data, writer=None, **kwargs):
        self.args = args
        self.data = data
        self.model = model
        self.writer = writer
        self.len = len(data['train']['edge_index_list'])
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        x = data['x'].to(args.device)
        self.x = [x for _ in range(self.len)] if len(x.shape) <= 2 else x
        setup_seed(args.seed)
        print('total length: {}, test length: {}'.format(
            self.len, args.testlength))

    def run(self):
        args = self.args
        max_auc = 0
        max_test_auc = 0
        max_train_auc = 0
        min_epoch = args.min_epoch
        max_patience = args.patience

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        t_total0 = time.time()
        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                average_epoch_loss, average_train_auc, average_val_auc, average_test_auc = self.train(epoch, self.data['train'])

                # update the best results.
                if average_val_auc > max_auc:
                    max_auc = average_val_auc
                    max_test_auc = average_test_auc
                    max_train_auc = average_train_auc

                    # ood results
                    test_results = self.test(epoch, self.data['test'])
                    patience = 0
                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break

                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print(
                        "Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(epoch, average_epoch_loss, time.time() - t0))
                    print(
                        f"(IID) Current: Epoch:{epoch}, Train AUC :{average_train_auc:.4f}, Val AUC_s: {average_val_auc:.4f}, Test AUC_s: {average_test_auc:.4f}"
                    )
                    print(
                        f"(IID) Best_Test: Epoch:{test_results[0]}, Train AUC:{max_train_auc:.4f}, Val AUC: {max_auc:.4f}, Test AUC: {max_test_auc:.4f}"
                    )
                    print(
                        f"(OOD) Current_Test: Epoch:{test_results[0]}, Train AUC:{test_results[1]:.4f}, Val AUC: {test_results[2]:.4f}, Test AUC: {test_results[3]:.4f}"
                    )

        epoch_time = (time.time() - t_total0) / (epoch - 1)
        return max_train_auc, max_auc, max_test_auc, test_results,  epoch_time, epoch, average_train_auc, average_val_auc, average_test_auc

    def train(self, epoch, data):
        self.model.train()
        optimizer = self.optimizer
        causal_embedding_list,  conf_embedding_list, env_kl_loss = \
            self.model(data['edge_index_list'], self.x, self.len_train)
        conf_embedding_list = [emb.detach() for emb in conf_embedding_list]

        causal_loss = torch.tensor([]).to(self.args.device)
        env_loss = torch.tensor([]).to(self.args.device)
        for t in range(self.len_train-1):
            causal_embedding = causal_embedding_list[:, t, :].squeeze() #[N, F]

            if self.args.dataset == 'act':
                pos_edge_index = data['pedges'][t + 1].long().to(self.args.device)
                neg_edge_index = negative_sampling(pos_edge_index, self.args.num_nodes, num_neg_samples= pos_edge_index.shape[1])

            else:
                pos_edge_index, neg_edge_index = data['pedges'][t+1].long().to(self.args.device),data['nedges'][t+1].long().to(self.args.device)

            causal_pos_pred, causal_neg_pred = self.cal_pred(causal_embedding, pos_edge_index, neg_edge_index,
                                                                     self.model.cs_decoder)
            causal_loss = torch.cat([causal_loss, self.loss(causal_pos_pred, causal_neg_pred).unsqueeze(0)])

            # causal intervention
            for times in range(self.args.intervention_times):
                n_s = np.random.randint(self.len_train)
                conf_z = conf_embedding_list[n_s]
                conf_pos_pred, conf_neg_pred = self.cal_pred(conf_z, pos_edge_index, neg_edge_index,
                                                             self.model.ss_decoder)
                s1 = np.random.randint(len(conf_pos_pred))
                conf_pos_pred_s = torch.sigmoid(conf_pos_pred[s1]).detach()
                conf_neg_pred_s = torch.sigmoid(conf_neg_pred[s1]).detach()
                conf_pos = conf_pos_pred_s * causal_pos_pred
                conf_neg = conf_neg_pred_s * causal_neg_pred
                env_loss = torch.cat([env_loss, self.loss(conf_pos, conf_neg).unsqueeze(0)])
        env_mean = env_loss.mean()
        env_var = torch.var(env_loss * self.args.intervention_times)
        penalty = env_mean + env_var

        la = self.args.weight1
        if epoch < self.args.warm_epoch:
            la = 0
        loss = torch.mean(causal_loss) + la * penalty + self.args.weight2 *torch.mean(env_kl_loss)
        # loss = torch.mean(causal_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_epoch_loss = loss.item()

        # train_test(iid data)
        self.model.eval()
        train_auc_list,  val_auc_list, test_auc_list = [], [], []
        causal_embedding_list, _, _ = \
            self.model(data['edge_index_list'], self.x, self.len)
        for t in range(self.len - 1):
            z_mix = causal_embedding_list[:, t, :].squeeze() #[N, F]
            pos_edge, neg_edge = data['pedges'][t+1].long().to(self.args.device),data['nedges'][t+1].long().to(self.args.device)
            auc_mix, ap_mix = self.predict(z_mix, pos_edge, neg_edge, self.model.cs_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc_mix)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc_mix)
            else:
                test_auc_list.append(auc_mix)

        return average_epoch_loss, np.mean(train_auc_list), np.mean(val_auc_list), np.mean(test_auc_list)


    def test(self, epoch, data):
        self.model.eval()
        causal_embedding_list, _, _ = self.model(data['edge_index_list'], self.x, self.len)
        train_auc_list, val_auc_list, test_auc_list = [], [], []
        for t in range(self.len - 1):
            z_mix = causal_embedding_list[:, t, :].squeeze() #[N, F]

            pos_edge, neg_edge = data['pedges'][t + 1].long().to(self.args.device), data['nedges'][t + 1].long().to(
                self.args.device)
            if neg_edge.shape[1]==0:
                continue
            auc_mix, ap_mix = self.predict(z_mix, pos_edge, neg_edge, self.model.cs_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc_mix)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc_mix)
            else:
                test_auc_list.append(auc_mix)

        return epoch, np.mean(train_auc_list), np.mean(val_auc_list), np.mean(test_auc_list)

    def cal_pred(self, z, edge_index_pos, edge_index_neg, decoder):

        pos_pred = decoder(z, edge_index_pos)
        neg_pred = decoder(z, edge_index_neg)
        return pos_pred, neg_pred

    def loss(self, pos_pred, neg_pred):
        pos_loss = -torch.log(pos_pred + 1e-9).mean()
        neg_loss = -torch.log(1 - neg_pred + 1e-9).mean()

        return pos_loss + neg_loss

    def predict(self, z, pos_edge_index, neg_edge_index, decoder):
        pos_y = z.new_ones(pos_edge_index.size(1)).to(z.device)
        neg_y = z.new_zeros(neg_edge_index.size(1)).to(z.device)
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = decoder(z, pos_edge_index)
        neg_pred = decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

