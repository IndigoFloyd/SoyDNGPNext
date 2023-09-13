import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
from soydngpnext.reader_cpu import Reader_CPU, one_hot_CPU
path = os.path.dirname(__file__)

# In this module, .csv file would be read and split into test and train datasets.
# A .yaml config file should be provided. Due to randomly split, datasets might be different.
class DataProcess:

    def __init__(self, genotype_file, trait_file):
        # path of the .vcf genotype file
        self.genotype_path = genotype_file
        # path of the .csv traits file
        self.trait_path = trait_file
        # the trait file DataFrame, be assigned after running method convert_trait
        self.trait = None
        # the SNP IDs
        self.columns = []

    # extract trait value
    def convert_trait(self, save_path):
        # the .csv file saves traits and data, for example:
        # acid SCN3 protein
        #  id1   1      1
        #  idn   n      n
        # load the csv file
        df = pd.read_csv(self.trait_path, encoding=u'gbk')
        # load the traits config file, for example
        # p:
        #  protein
        # n:
        #  SN3
        with open(f'{path}/data/traits.yaml') as t:
            traits = yaml.safe_load(t)
        # p_trait_dict: {'trait1': {0:level a, 1:level b...}...}
        # n_trait_dict: {'trait1': {max:value1, min:value2}...}
        p_trait_dict = {}
        n_trait_dict = {}
        # the Quality Traits
        if traits['p']:
            quality = df[traits['p']]
            # keep different levels of this trait
            for column in quality.columns:
                # avoid empty value to be added as a level
                tmp = df[column].dropna().drop_duplicates()
                trait_dict = tmp.reset_index(drop=True).to_dict()
                swapped_dict = {value: key for key, value in trait_dict.items()}
                df[column] = df[column].replace(swapped_dict)
                p_trait_dict[column] = trait_dict
            del tmp

        # the Quantity Traits
        if traits['n']:
            quantity = df[traits['n']]
            n_min = quantity.min()
            n_max = quantity.max()
            # normalize the input data for training
            df[traits['n']] = (quantity - n_min) / (n_max - n_min)
            # keep the information of max value and minimum value of this trait for denormalization
            for trait in n_min.index:
                n_trait_dict[trait] = {"max": float(n_max[trait]), "min": float(n_min[trait])}
        # save dictionaries as .yaml files
        os.makedirs(save_path, exist_ok=True)
        with open(f'{save_path}/n_trait.yaml', 'w') as n_file:
            n_file.write(yaml.dump(n_trait_dict))
        with open(f'{save_path}/p_trait.yaml', 'w') as p_file:
            p_file.write(yaml.dump(p_trait_dict))
        self.trait = df.set_index('acid')
        return p_trait_dict, n_trait_dict

    # split dataset into train and test dataset, and encode genotype data by one-hot method
    def to_dataset(self, trait_to_train, percentage, is_quality):
        # bool list of non-missing values
        mask = self.trait[trait_to_train].notna()
        # IDs of non-missing values
        IDs = self.trait.index[mask].tolist()
        # ignore missing values, and transform to numpy array
        trait_value = self.trait[trait_to_train][mask].values
        r = Reader_CPU()
        # read the .vcf genotype file
        genotype_original = r.readVCF(self.genotype_path)
        # the SNP IDs
        self.columns = genotype_original.columns.to_list()
        # ignore missing values, and transform to numpy array
        genotype_value = genotype_original.loc[IDs, :].values
        # if trait_to_train is a quality trait, then split each level by the same percentage
        # finally, make genotype data one-hot encoded
        if is_quality:
            train_data, test_data, train_label, test_label = train_test_split(genotype_value, trait_value, train_size=percentage, stratify=trait_value)
            train_data = one_hot_CPU(train_data)
            test_data = one_hot_CPU(test_data)
        else:
            train_data, test_data, train_label, test_label = train_test_split(genotype_value, trait_value, train_size=percentage)
            train_data = one_hot_CPU(train_data)
            test_data = one_hot_CPU(test_data)
        return train_data, train_label, test_data, test_label

# an example
# d = DataProcess("../data/test.vcf", "../data/final_trait.csv")
# d.convert_trait()

