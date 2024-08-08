from data_utils import load_data
from logger import Logger
from config import args
import os
import os.path as osp
from trainner import Trainer
from trainner_nc import NCTrainer
from trainner_motif import NC_Motif_Trainer
from model import DyCIL
import numpy as np
args, data = load_data(args)
exp_dir = osp.join('log/', args.experiment_name, args.dataset, str(args.P))
os.makedirs(exp_dir, exist_ok=True)
logger = Logger.init_logger(filename=exp_dir + '/DyCIL.log')
logger.info(args)


train_list, val_list, test_list= [], [], []
train_ood_list, val_ood_list, test_ood_list = [], [], []
test15, test16, test17 = [],[],[]
for i in range(3):
    model = DyCIL(args).to(args.device)
    if args.experiment_name == 'lp':
        trainner = Trainer(args, model, data)
        results = trainner.run()
        logger.info(
            f"(IID) Last_Train: Epoch:{results[5]}, Train AUC:{results[6]:.4f}, Val AUC: {results[7]:.4f}, Test AUC: {results[8]:.4f}")
        logger.info(
            f"(IID) Best_Train: Epoch:{results[3][0]}, Train AUC:{results[0]:.4f}, Val AUC: {results[1]:.4f}, Test AUC: {results[2]:.4f}")
        logger.info(
            f"(OOD) Early_Test: Epoch:{results[3][0]}, Train AUC:{results[3][1]:.4f}, Val AUC: {results[3][2]:.4f}, Test AUC: {results[3][3]:.4f}")
        logger.info(f'running time per every epoch: {results[4]:.4f}')
        # iid
        train_list.append(results[0])
        val_list.append(results[1])
        test_list.append(results[2])
        # ood
        train_ood_list.append(results[3][1])
        val_ood_list.append(results[3][2])
        test_ood_list.append(results[3][3])
    else:
        if args.dataset == 'Aminer':
            trainner = NCTrainer(args, model, data)
            results = trainner.run()
            logger.info(f"Last_Train: Epoch:{results[0]}, Train ACC:{results[1]:.4f}, Val ACC: {results[2]:.4f}, Test ACC: {results[3]:.4f}")
            logger.info(f"Best_Val: Epoch:{results[4]}, Train ACC:{results[5]:.4f}, Val ACC: {results[6]:.4f}, Test ACC: {results[7]:.4f}")
            logger.info(f"Every Test: Epoch:{results[4]}, Test15:{results[9][0]:.4f}, Test16: {results[9][1]:.4f}, Test17: {results[9][2]:.4f}")
            logger.info(f'running time per every epoch: {results[8]:.4f}')
            # ood
            train_list.append(results[5])
            val_list.append(results[6])
            test_list.append(results[7])
            # every test acc
            test15.append(results[9][0])
            test16.append(results[9][1])
            test17.append(results[9][2])
        elif args.dataset == 'dymotif_data':
            trainner = NC_Motif_Trainer(args, model, data)
            results = trainner.run()
            logger.info(f"Last_Train: Epoch:{results[0]}, Train ACC:{results[1]:.4f}, Val ACC: {results[2]:.4f}, Test ACC: {results[3]:.4f}")
            logger.info(f"Best_Val: Epoch:{results[4]}, Train ACC:{results[5]:.4f}, Val ACC: {results[6]:.4f}, Test ACC: {results[7]:.4f}")
            logger.info(f"Every Test: Epoch:{results[4]}, Test15:{results[9][0]:.4f}, Test16: {results[9][1]:.4f}, Test17: {results[9][2]:.4f}")
            logger.info(f'running time per every epoch: {results[8]:.4f}')
            # ood
            train_list.append(results[5])
            val_list.append(results[6])
            test_list.append(results[7])
            # every test acc
            test15.append(results[9][0])
            test16.append(results[9][1])
            test17.append(results[9][2])

        # elif args.dataset == 'synthetic_SBM':
        #     trainner = NC_SBM_Trainer(args, model, data)
        #     results = trainner.run()
        #     logger.info(f"Last_Train: Epoch:{results[0]}, Train ACC:{results[1]:.4f}, Val ACC: {results[2]:.4f}, Test ACC: {results[3]:.4f}")
        #     logger.info(f"Best_Val: Epoch:{results[4]}, Train ACC:{results[5]:.4f}, Val ACC: {results[6]:.4f}, Test ACC: {results[7]:.4f}")
        #     logger.info(f'running time per every epoch: {results[8]:.4f}')
        #     # ood
        #     train_list.append(results[4])
        #     val_list.append(results[5])
        #     test_list.append(results[6])

if args.experiment_name == 'lp':
    train_mean, train_std, val_mean, val_std, test_mean, test_std = \
        np.mean(train_list), np.std(train_list), np.mean(val_list), np.std(val_list), np.mean(test_list), np.std(test_list)
    train_ood_mean, train_ood_std, val_ood_mean, val_ood_std, test_ood_mean, test_ood_std = \
        np.mean(train_ood_list), np.std(train_ood_list), np.mean(val_ood_list), np.std(val_ood_list), np.mean(test_ood_list), np.std(test_ood_list)
    logger.info(
        f"(IID Mean Results) Train:{train_mean*100:.2f}±{train_std*100:.2f}, Val: {val_mean*100:.2f}±{val_std*100:.2f}, Test: {test_mean*100:.2f}±{test_std*100:.2f}")
    logger.info(
        f"(OOD Mean Results) Train:{train_ood_mean * 100:.2f}±{train_ood_std * 100:.2f}, Val: {val_ood_mean * 100:.2f}±{val_ood_std * 100:.2f}, Test: {test_ood_mean * 100:.2f}±{test_ood_std * 100:.2f}")
else:
    train_mean, train_std, val_mean, val_std, test_mean, test_std = \
        np.mean(train_list), np.std(train_list), np.mean(val_list), np.std(val_list), np.mean(test_list), np.std(
            test_list)
    test15_mean, test15_std, test16_mean, test16_std, test17_mean, test17_std = \
        np.mean(test15), np.std(test15), np.mean(test16), np.std(test16), np.mean(test17), np.std(test17)
    # train_ood_mean, train_ood_std, val_ood_mean, val_ood_std, test_ood_mean, test_ood_std = \
    #     np.mean(train_ood_list), np.std(train_ood_list), np.mean(val_ood_list), np.std(val_ood_list), np.mean(
    #         test_ood_list), np.std(test_ood_list)
    logger.info(
        f"(Mean Results) Train:{train_mean * 100:.2f}±{train_std * 100:.2f}, Val: {val_mean * 100:.2f}±{val_std * 100:.2f}, Test: {test_mean * 100:.2f}±{test_std * 100:.2f}")
    logger.info(
        f"(Every Results) Train:{test15_mean * 100:.2f}±{test15_std * 100:.2f}, Val: {test16_mean * 100:.2f}±{test16_std * 100:.2f}, Test: {test17_mean * 100:.2f}±{test17_std * 100:.2f}")


# model = DyCIL(args).to(args.device)
# if args.experiment_name == 'lp':
#     trainner = Trainer(args, model, data)
#     results = trainner.run()
#     logger.info(
#         f"(IID) Last_Train: Epoch:{results[5]}, Train AUC:{results[6]:.4f}, Val AUC: {results[7]:.4f}, Test AUC: {results[8]:.4f}")
#     logger.info(
#         f"(IID) Best_Train: Epoch:{results[3][0]}, Train AUC:{results[0]:.4f}, Val AUC: {results[1]:.4f}, Test AUC: {results[2]:.4f}")
#     logger.info(
#         f"(OOD) Early_Test: Epoch:{results[3][0]}, Train AUC:{results[3][1]:.4f}, Val AUC: {results[3][2]:.4f}, Test AUC: {results[3][3]:.4f}")
#     logger.info(f'running time per every epoch: {results[4]:.4f}')
# else:
#     if args.dataset == 'Aminer':
#         trainner = NCTrainer(args, model, data)
#         results = trainner.run()
#         logger.info(f"Last_Train: Epoch:{results[0]}, Train ACC:{results[1]:.4f}, Val ACC: {results[2]:.4f}, Test ACC: {results[3]:.4f}")
#         logger.info(f"Best_Val: Epoch:{results[4]}, Train ACC:{results[5]:.4f}, Val ACC: {results[6]:.4f}, Test ACC: {results[7]:.4f}")
#         logger.info(f"Every Test: Epoch:{results[4]}, Test15:{results[9][0]:.4f}, Test16: {results[9][1]:.4f}, Test17: {results[9][2]:.4f}")
#         logger.info(f'running time per every epoch: {results[8]:.4f}')
#
#     elif args.dataset == 'dymotif_data':
#         trainner = NC_Motif_Trainer(args, model, data)
#         results = trainner.run()
#         logger.info(f"Last_Train: Epoch:{results[0]}, Train ACC:{results[1]:.4f}, Val ACC: {results[2]:.4f}, Test ACC: {results[3]:.4f}")
#         logger.info(f"Best_Val: Epoch:{results[4]}, Train ACC:{results[5]:.4f}, Val ACC: {results[6]:.4f}, Test ACC: {results[7]:.4f}")
#         logger.info(f"Every Test: Epoch:{results[4]}, Test28:{results[9][0]:.4f}, Test29: {results[9][1]:.4f}, Test30: {results[9][2]:.4f}")
#         logger.info(f'running time per every epoch: {results[8]:.4f}')


