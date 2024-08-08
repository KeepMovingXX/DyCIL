import time
import torch
import numpy as np
from tqdm import tqdm
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class NCTrainer(object):
    def __init__(self, args, model, data, writer=None, **kwargs):
        self.args = args
        self.data = data
        self.model = model
        self.writer = writer
        self.len = len(data['edge_index'])
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
        max_acc = 0
        max_test_acc = 0
        max_train_acc = 0
        min_epoch = args.min_epoch
        max_patience = args.patience

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        t_total0 = time.time()
        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                average_epoch_loss, average_train_acc, average_val_acc, average_test_acc, test_acc_list = self.train(epoch, self.data)
                # update the best results.
                if average_val_acc > max_acc:
                    max_acc = average_val_acc
                    max_test_acc = average_test_acc
                    max_train_acc = average_train_acc
                    best_test_acc_list = test_acc_list
                    patience = 0
                    best_epoch = epoch
                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break

                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print(
                        "Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(epoch, average_epoch_loss, time.time() - t0))
                    print(
                        f"Current: Epoch:{epoch}, Train AUC :{average_train_acc:.4f}, Val AUC: {average_val_acc:.4f}, Test AUC: {average_test_acc:.4f}"
                    )
                    print(
                        f"Best_metric: Epoch:{best_epoch}, Train AUC:{max_train_acc:.4f}, Val AUC: {max_acc:.4f}, Test AUC: {max_test_acc:.4f}"
                    )
                    print(
                        f"Every Test: Epoch:{best_epoch}, Test15:{best_test_acc_list[0]:.4f}, Test16: {best_test_acc_list[1]:.4f}, Test17: {best_test_acc_list[2]:.4f}")

        epoch_time = (time.time() - t_total0) / (epoch - 1)
        return epoch, average_train_acc, average_val_acc, average_test_acc, best_epoch, max_train_acc, max_acc, max_test_acc, epoch_time, best_test_acc_list

    def train(self, epoch, data):
        self.model.train()
        optimizer = self.optimizer

        causal_embedding_list, conf_embedding_list, env_kl_loss = \
            self.model(data['edge_index'], self.x[:self.len_train], self.len_train)
        conf_embedding_list = [emb.detach() for emb in conf_embedding_list]

        criterion = torch.nn.CrossEntropyLoss()
        causal_loss = torch.tensor([]).to(self.args.device)
        env_loss = torch.tensor([]).to(self.args.device)

        for t in range(self.len_train):
            causal_embedding = causal_embedding_list[:, t, :].squeeze() #[N, F]
            causal_pred = self.cal_pred(causal_embedding, self.model.cs_decoder, data['node_masks'][t].to(self.args.device))
            causal_loss = torch.cat([causal_loss, criterion(causal_pred, data['y'][data['node_masks'][t]].squeeze().to(self.args.device)).unsqueeze(0)])
            # causal intervention
            for times in range(self.args.intervention_times):
                n_s = np.random.randint(self.len_train)
                conf_z = conf_embedding_list[n_s]
                conf_pred = self.cal_pred(conf_z, self.model.ss_decoder, data['node_masks'][t].to(self.args.device))
                s1 = np.random.randint(len(conf_pred))
                conf_pred_s = torch.sigmoid(conf_pred[s1]).detach()
                conf = conf_pred_s * causal_pred
                env_loss = torch.cat([env_loss, criterion(conf, data['y'][data['node_masks'][t]].squeeze().to(self.args.device)).unsqueeze(0)])
        env_mean = env_loss.mean()
        env_var = torch.var(env_loss * self.args.intervention_times *self.len_train)
        penalty = env_mean + env_var

        la = self.args.weight1
        if epoch < self.args.warm_epoch:
            la = 0
        loss = torch.mean(causal_loss) + la * penalty + self.args.weight2 *torch.mean(env_kl_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_epoch_loss = loss.item()

        # get acc
        self.model.eval()
        train_acc_list,  val_acc_list, test_acc_list = [], [], []
        causal_embedding_list, _, _  = self.model(data['edge_index'], self.x, self.len)
        for t in range(self.len):
            z = causal_embedding_list[:, t, :].squeeze() #[N, F]
            acc = self.predict(z, self.model.cs_decoder, data['node_masks'][t], data['y'])
            if t < self.len_train:
                train_acc_list.append(acc)
            elif t < self.len_train + self.len_val:
                val_acc_list.append(acc)
            else:
                test_acc_list.append(acc)

        return average_epoch_loss, np.mean(train_acc_list), np.mean(val_acc_list), np.mean(test_acc_list), test_acc_list


    def cal_pred(self, z, decoder, node_masks):
        pred = decoder(z)[node_masks]
        return pred

    def predict(self, z, decoder, node_mask, y):
        pred = decoder(z)[node_mask]
        pred = pred.argmax(dim=-1).squeeze()
        y = y[node_mask].squeeze()
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        acc = (pred == y).sum().item() / y.shape[0]
        acc = float(acc)
        # acc = accuracy_score(y, pred)
        return acc

