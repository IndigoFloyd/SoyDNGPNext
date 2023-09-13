import pandas as pd
import numpy as np

# This module is a CPU implement of reader.py

class Reader_CPU:
    # initial class Reader, load the needed SNP file
    def __init__(self):
        self.columns = []
        self.indexes = None

    # the CPU implement of readVCF method
    def readVCF(self, vcf_path, reset=False):
        # open the .vcf file
        with open(vcf_path, "r") as f:
            # if the line does not start as "##", this line is the header line
            header = next((i for i, line in enumerate(f) if not line.startswith('##')), 0)
        # read the file from header row
        df = pd.read_csv(vcf_path, header=header, sep='\t', dtype=str)
        # clean the useless columns, such as ['QUAL', 'FILTER', 'INFO', 'FORMAT', '#CHROM', 'POS', 'REF', 'ALT']
        # after doing that, only ID and sample IDs will be kept
        self.columns.append(df.columns[2])
        self.columns.extend(df.columns[9:])
        # concat the chromosome name and position as new ID for SNPs
        df['ID'] = df['#CHROM'].str.cat(df['POS'], sep='_')
        # set ID column as index and drop the useless columns
        df = df[self.columns].set_index('ID')
        # clear values of DataFrame
        # sometimes, values might be with annotations, for example, '1/1(some annotation)', which should be deleted
        for column in self.columns[1:]:
            # only the first three characters could be kept
            df[column] = df[column].str.slice(0, 3)
            # create a temporary series, which is replaced by rules
            temp_series = df[column].replace({r'1/1': '1', r'1|1': '1', r'0/1': '2', r'0|1': '2'})
            # make a mask to find out unchanged elements
            mask = df[column] == temp_series
            # assign '3' for unchanged elements
            temp_series[mask] = '3'
            # replace column of DataFrame with assigned series
            df[column] = temp_series.astype(np.int32)
            # delete useless variables
            del temp_series, mask
        # transpose the DataFrame, now the SNP IDs are columns, and sample IDs are indexes
        df = df.transpose()
        # keep the indexes and columns
        self.indexes = df.index
        self.columns = df.columns
        # drop indexes and reduce memory consumption
        if reset:
            df.reset_index(drop=True, inplace=True)
        return df.astype(np.int32)


# the CPU implement of one_hot method
def one_hot_CPU(matrix):
    # repeat 3 times to 3D, in this way, transforming to one-hot matrix does not need explicit loop, which could perform better
    matrix_3D = np.zeros((matrix.shape[0], matrix.shape[1], 3))
    # the rule for one-hot is like:
    #             '1/1' or '1|1'        '0/1' or '0|1'        other
    # channel 1          1                     1                0
    # channel 2          1                     0                1
    # channel 3          0                     1                1
    # for the first layer, only the first and second condition will be valued as 1
    matrix_3D[:, :, 0] = np.where(np.isin(matrix[:, :], np.array([1, 2])), 1, 0)
    # for the second layer, only the second condition will be valued as 0
    matrix_3D[:, :, 1] = np.where(np.isin(matrix[:, :], np.array(2)), 0, 1)
    # for the third layer, only the first condition will be valued as 0
    matrix_3D[:, :, 2] = np.where(np.isin(matrix[:, :], np.array(1)), 0, 1)
    # resize every sample to shape of model input
    # set an index
    n = -1
    # create batches of resized matrix
    sample_resized = np.empty((matrix.shape[0], 3, 206, 206)).astype(np.float32)
    for sample in matrix_3D:
        # !!!!!(cite from numpy v1.25 document)
        # If the new array is larger than the original array, then the new array is filled with repeated copies of a.
        # Note that this behavior is different from a.resize(new_shape) which fills with zeros instead of repeated copies of a.
        # !!!!!
        sample = np.resize(sample, (206, 206, 3))
        # transpose the order of sample's channels, for cp.reshape might change its order
        # for example, before reshape, sample has order [a, b, c], which might be changed after reshape, like [1, c, b, a]
        # to keep the result's order, here a transposition is significant
        sample = np.transpose(sample, (2, 0, 1))
        # the first dimension is batch size, just for fitting the input shape
        sample = np.reshape(sample, (1, 3, 206, 206))
        n += 1
        sample_resized[n, :, :, :] = sample
    return sample_resized
