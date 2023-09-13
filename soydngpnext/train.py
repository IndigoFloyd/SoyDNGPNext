import warnings
import torch
from torch.utils.data import DataLoader
# from SoyDNGPNext.weight_map import weight_decoder
from soydngpnext.remodel import remodel
from sklearn import metrics
from soydngpnext.data_process import *
from soydngpnext.utils import outpath
from soydngpnext.eval import *
import os
path = os.path.dirname(__file__)

def init_weights(net):
    if type(net) == torch.nn.Linear or type(net) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(net.weight)


class data_loader(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        genotype = self.data[index].float()
        label = self.label[index].float()
        return genotype, label


# In this module, users could train their own datasets on our baseline
class Train:
    def __init__(self, vcf_path, trait_path, percentage_of_train=0.7, num_workers=8, batch_size=20):
        # ignore warnings
        warnings.filterwarnings("ignore")
        # select which device to train
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # percentage of train dataset
        self.percentage_of_train = percentage_of_train
        # the training batch size
        self.batch_size = batch_size
        # workers amount
        self.num_workers = num_workers
        # output path
        self.saved_path = outpath('train')
        # read .vcf file and .csv trait file, and instantiate the data process class
        self.d = DataProcess(vcf_path, trait_path)
        # process and return the Quality Traits dictionary and Quantity Traits dictionary
        self.p_trait_dic, self.n_trait_dic = self.d.convert_trait(f"{self.saved_path}/configs/")
        print(f"yaml configs are saved in {self.saved_path}/configs/")
        # turn keys to lists
        self.p_trait_list, self.n_trait_list = list(self.p_trait_dic.keys()), list(self.n_trait_dic.keys())



    # create train and test dataloader for training
    def dataloader(self, trait, is_quality):
        # split dataset by percentage of dataset
        train_data, train_label, test_data, test_label = self.d.to_dataset(trait, percentage=self.percentage_of_train, is_quality=is_quality)
        # dataloader of train dataset
        train_dataloader = DataLoader(data_loader(train_data, train_label), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        # dataloader of test dataset
        test_dataloader = DataLoader(data_loader(test_data, test_label), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return train_dataloader, test_dataloader

    # train an epoch
    def train_for_epoch(self, train_dataloader, updater, loss, net):
        loss_ = 0.0
        for num_data, (genomap, target_trait) in enumerate(train_dataloader):
            genomap, target_trait = genomap.to(self.device), target_trait.to(self.device)
            trait_hat = net(genomap)
            loss_for_batch = loss(trait_hat, target_trait.long())
            loss_ += loss_for_batch
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                loss_for_batch.backward()
                updater.step()
        return loss_ / (num_data + 1)

    # train the Quantity Traits
    def train_n(self, percentage=0.05, epoch=5, weight_decay=1e-5, draw=True):
        epoch_total = epoch
        weight_save_path = f"{self.saved_path}/weight/"
        pic_save_path = f"{self.saved_path}/"
        os.makedirs(weight_save_path, exist_ok=True)
        net, decoder = remodel(f"{path}/data/model.yaml", 1)
        net.to(self.device)
        # initialize weights
        net.apply(init_weights)
        # set Smooth S1 Loss as the loss function
        loss = torch.nn.SmoothL1Loss()
        # set updater with Adam algorithm
        updater = torch.optim.Adam(net.parameters(), weight_decay=weight_decay)
        for trait in self.n_trait_list:
            # create the evaluation dictionary, Quantity Traits and Quality Traits have different evaluation indexes
            eval_dict = {'train_loss': [], 'test_loss': [], 'mse': [], 'r': [], }
            # initialize the loss function
            mse_loss = torch.nn.MSELoss()
            # train and test dataloaders
            train_dataloader, test_dataloader = self.dataloader(trait, is_quality=False)
            # train each epoch
            # the minimum coefficient value
            min_r = 0.0
            while epoch:
                print(f"Now training epoch {epoch_total - epoch + 1}")
                # set the model as train mode
                net.train()
                # train this epoch and compute the average loss
                avg_train_loss = self.train_for_epoch(train_dataloader, updater, loss, net)
                # set the model as evaluation mode
                net.eval()
                # for the test dataset
                with torch.no_grad():
                    # initialize the evaluation indexes
                    mse = 0.0
                    loss_test = 0.0
                    hat = np.array([])
                    truth = np.array([])
                    for index, (test_genomap, test_trait) in enumerate(test_dataloader):
                        # load the feature maps and traits on device
                        test_genomap, test_trait = test_genomap.to(self.device), test_trait.to(self.device)
                        # the predict results
                        y_hat = net(test_genomap)
                        # compute loss on test dataset
                        loss_test += loss(y_hat, test_trait.long())
                        # reshape y_hat as an 1-dimension vector, and insert it to hat
                        hat = np.insert(hat, 0, y_hat.to('cpu').detach().numpy().reshape(len(y_hat), ), axis=0)
                        # reshape acc as an 1-dimension vector, and insert it to acc
                        truth = np.insert(truth, 0, test_trait.to('cpu').numpy(), axis=0)
                        # compute MSE between predict results and actual values
                        mse += mse_loss(y_hat, test_trait)
                # count evaluation indexes
                loss_test = loss_test / (index + 1)
                eval_dict['test_loss'].append(float(loss_test.to('cpu').detach().numpy()))
                eval_dict['train_loss'].append(float(avg_train_loss.to('cpu').detach().numpy()))
                # the average MSE
                mse / (index + 1)
                # compute the correlation coefficient value
                r = np.corrcoef(hat, truth)[0][1]
                # append the coefficient value into evaluation dictionary
                eval_dict['r'].append(r)
                # save the best weight with the highest coefficient value
                if r >= min_r:
                    torch.save(net, os.path.join(weight_save_path, f'{trait}_best.pt'))
                    # refresh the minimum coefficient value
                    min_r = eval_dict['r'][-1]
                epoch -= 1
            print(f"weight are saved in {weight_save_path}/{trait}_best.pt")
            # set the trait of this evaluation dictionary
            eval_dict['trait'] = trait
            if draw:
                Eval(eval_dict, pic_save_path)
                print(f"pictures are saved in {pic_save_path}")
        # snp_pos = weight_decoder(net,decoder,self.d.columns, percentage=percentage)
        # print(f"Top {percentage*100}% most important SNP positions are: {snp_pos}")
        return eval_dict

    # train the Quality Traits
    def train_p(self, percentage=0.05, epoch=5, weight_decay=1e-5, draw=True):
        epoch_total = epoch
        weight_save_path = f"{self.saved_path}/weight/"
        pic_save_path = f"{self.saved_path}/"
        os.makedirs(weight_save_path, exist_ok=True)
        for trait in self.p_trait_list:
            # train and test dataloaders
            train_dataloader, test_dataloader = self.dataloader(trait, is_quality=True)
            # get how many levels this trait has
            num_classes = len(self.p_trait_dic[trait])
            # load the model
            # for different class amounts, the net has to be reloaded in each loop
            net, decoder = remodel(f"{path}/data/model.yaml", num_classes)
            net.to(self.device)
            # initialize weights
            net.apply(init_weights)
            # set Entropy Loss as the loss function
            loss = torch.nn.CrossEntropyLoss()
            # set updater with Adam algorithm
            updater = torch.optim.Adam(net.parameters(), weight_decay=weight_decay)
            # evaluation dictionary of Quality Traits
            eval_dict = {'train_loss': [], 'test_loss': [], 'acc': [], 'recall': [], 'precision': [], 'f1_score': []}
            # the minimum evaluation value
            eval_value = 0.0
            # train each epoch
            while epoch:
                print(f"Now training epoch {epoch_total - epoch + 1}")
                # set the model as train mode
                net.train()
                # train this epoch and compute the average loss
                avg_train_loss = self.train_for_epoch(train_dataloader, updater, loss, net)
                # set the model as evaluation mode
                net.eval()
                # for the test dataset
                with torch.no_grad():
                    # initialize the evaluation indexes
                    acc_score = 0.0
                    recall_score = 0.0
                    f1_score = 0.0
                    precision = 0.0
                    loss_test = 0.0
                    hat = np.array([])
                    truth = np.array([])
                    for index, (test_genomap, test_trait) in enumerate(test_dataloader):
                        # load the feature maps and traits on device
                        test_genomap, test_trait = test_genomap.to(self.device), test_trait.to(self.device)
                        # the predict results
                        y_hat = net(test_genomap)
                        # compute loss on test dataset
                        loss_test += loss(y_hat, test_trait.long())
                        # reshape y_hat as an 1-dimension vector, and insert it to hat
                        hat = np.insert(hat, 0, np.argmax(y_hat.to('cpu').detach().numpy(), axis=1), axis=0)
                        # reshape acc as an 1-dimension vector, and insert it to acc
                        truth = np.insert(truth, 0, test_trait.to('cpu').numpy(), axis=0)
                        # compute accuracy scores
                        acc_score += metrics.accuracy_score(truth, hat)
                        # compute recall scores
                        recall_score += metrics.recall_score(truth, hat, average='macro')
                        # compute F1 scores
                        f1_score += metrics.f1_score(truth, hat, average='macro')
                        # compute precision scores
                        precision += metrics.precision_score(truth, hat, average='macro')
                # count evaluation indexes
                loss_test = loss_test / (index + 1)
                eval_dict['test_loss'].append(float(loss_test.to('cpu').detach().numpy()))
                eval_dict['train_loss'].append(float(avg_train_loss.to('cpu').detach().numpy()))
                eval_dict['acc'].append(acc_score / (index + 1))
                eval_dict['recall'].append(recall_score / (index + 1))
                eval_dict['f1_score'].append(f1_score / (index + 1))
                eval_dict['precision'].append(precision / (index + 1))
                # save the best weight with the highest coefficient value
                value = np.array(list(eval_dict.values()))[2:, -1].mean()
                if value > eval_value:
                    # compute the confuse matrix
                    truth = list(map(int, truth))
                    hat = list(map(int, hat))
                    confusion_matrix = metrics.confusion_matrix(truth, hat, labels=list(self.p_trait_dic[trait].keys()), sample_weight=None)
                    # refresh the evaluation value
                    eval_value = value
                    torch.save(net, os.path.join(weight_save_path, f'{trait}_best.pt'))
                epoch -= 1
            print(f"weight are saved in {weight_save_path}/{trait}_best.pt")
            # set the trait of this evaluation dictionary and save the confusion matrix
            eval_dict['trait'] = trait
            eval_dict['confusion_matrix'] = confusion_matrix
            if draw:
                Eval(eval_dict, pic_save_path, self.p_trait_dic)
                print(f"pictures are saved in {pic_save_path}")
        # snp_pos = weight_decoder(net, decoder,self.d.columns, percentage=percentage)
        # print(f"Top {percentage * 100}% most important SNP positions are: {snp_pos}")
        return eval_dict


# an example
# t = Train("data/train_example.vcf", "data/train_example.csv")
# t.train_p(epoch=2)
