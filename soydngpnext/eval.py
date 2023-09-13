import matplotlib.pyplot as plt
import numpy as np
import yaml
import os


# In this module, the evaluation dictionary after running train method will be drawn.
class Eval:
    def __init__(self, eval_dict, save_dir, p_trait_dic=None):
        self.eval_dict = eval_dict
        self.save_dir = save_dir
        self.p_trait_dic = p_trait_dic
        self.model_eval()
        self.pic_draw()

    def model_eval(self):
        dict_ = {}
        trait = self.eval_dict['trait']
        if not self.p_trait_dic:
            best_model_index = self.eval_dict['r'].index(max(self.eval_dict['r']))
            for key in self.eval_dict.keys():
                dict_[key] = self.eval_dict[key][best_model_index]
            with open(f"{self.save_dir}configs/{trait}_module_eval.yaml", 'w') as f:
                f.write(yaml.dump(dict_, allow_unicode=True))
        else:
            temp_dict = dict([(key, self.eval_dict[key]) for key in ['acc', 'precision', 'recall', 'f1_score']])
            eval_data = np.array(list(temp_dict.values()))
            eval_data = np.sum(eval_data, axis=0)
            best_model_index = np.argmax(eval_data)
            for key in self.eval_dict.keys():
                if key != 'confusion_matrix' and key != 'trait':
                    dict_[key] = self.eval_dict[key][best_model_index]
                else:
                    dict_[key] = self.eval_dict[key]
            with open(f"{self.save_dir}configs/{trait}_module_eval.yaml", 'w') as f:
                f.write(yaml.dump(dict_, allow_unicode=True))

    def pic_draw(self):
        x = [i for i in range(1, len(self.eval_dict['train_loss']) + 1)]
        train_loss = self.eval_dict['train_loss']
        test_loss = self.eval_dict['test_loss']
        trait = self.eval_dict['trait']
        if not self.p_trait_dic:
            mse = self.eval_dict['mse']
            r = self.eval_dict['r']

            # loss curve
            plt.plot(x, train_loss, label='train', linewidth=0.9)
            plt.plot(x, test_loss, label='test', linewidth=0.9)
            plt.xlabel('Epoch', fontsize=13)
            plt.ylabel('SmoothL1 Loss', fontsize=13)
            plt.legend(loc='best')
            plt.savefig(os.path.join(self.save_dir, f'{trait}_loss.svg'))
            plt.close()
            # mse curve
            plt.plot(x, mse, linewidth=0.9)
            plt.xlabel('Epoch', fontsize=13)
            plt.ylabel('Meansquare Error', fontsize=13)
            plt.savefig(os.path.join(self.save_dir, f'{trait}_mse.svg'))
            plt.close()
            # coefficient curve
            plt.plot(x, r, linewidth=0.9)
            plt.xlabel('Epoch', fontsize=13)
            plt.ylabel('Pearson Correlation Coefficient', fontsize=13)
            plt.savefig(os.path.join(self.save_dir, f'{trait}_r.svg'))
            plt.close()

        else:
            acc = self.eval_dict['acc']
            pre = self.eval_dict['precision']
            recall = self.eval_dict['recall']
            f1_score = self.eval_dict['f1_score']
            confusion_matrix = self.eval_dict['confusion_matrix']

            # loss curve
            plt.plot(x, train_loss, label='train', linewidth=0.9)
            plt.plot(x, test_loss, label='test', linewidth=0.9)
            plt.xlabel('Epoch', fontsize=13)
            plt.ylabel('CrossEntrpy Loss', fontsize=13)
            plt.legend(loc='best')
            plt.savefig(os.path.join(self.save_dir, f'{trait}_loss.svg'))
            plt.close()
            # eval curve
            plt.plot(x, acc, label='accuracy', linewidth=0.9)
            plt.plot(x, pre, label='precision', linewidth=0.9)
            plt.plot(x, f1_score, label='f1_score', linewidth=0.9)
            plt.plot(x, recall, label='recall', linewidth=0.9)
            plt.xlabel('Epoch', fontsize=13)
            plt.ylabel('Values', fontsize=13)
            plt.legend(loc='best')
            plt.savefig(os.path.join(self.save_dir, f'{trait}_eval.svg'))
            plt.close()

            print(confusion_matrix)
            # confusion matrix
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            print(confusion_matrix)
            x = list(self.p_trait_dic[trait].keys())
            plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
            plt.xticks(x, list(self.p_trait_dic[trait].values()), rotation=45)
            plt.yticks(x, list(self.p_trait_dic[trait].values()))
            plt.ylabel('True label', fontsize=13)
            plt.xlabel('Predicted label', fontsize=13)
            for i in range(np.shape(confusion_matrix)[0]):
                for j in range(np.shape(confusion_matrix)[1]):
                    if int(confusion_matrix[i][j] * 100 + 0.5) > 0:
                        plt.text(j, i, format(int(confusion_matrix[i][j] * 100 + 0.5), 'd') + '%',
                                 ha="center", va="center",
                                 color="white" if confusion_matrix[i][j] > 0.7 else "black")
            plt.savefig(os.path.join(self.save_dir, f'{trait}_confusion_matrix.svg'))
            plt.close()
