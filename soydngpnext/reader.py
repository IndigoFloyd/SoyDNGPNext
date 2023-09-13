import cupy as cp
import cudf

# This module is used for read .vcf file, and get filtered SNP data for model input.
# A test .vcf file is provided in example/10_test_examples.vcf. Except contigs, the rest is read as a "\t" delimited csv file.
# The file is like:

# ......
# ##contig=<ID=Chr20>
# ##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes">
# ##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">
# ##bcftools_viewVersion=1.8+htslib-1.8
# ##bcftools_viewCommand=view -S test_sample.txt soybean.recode.vcasf.gz; Date=Sun Apr 30 21:11:24 2023
# ##bcftools_viewVersion=1.15.1-26-geafc742+htslib-1.15.1-55-gb7addd3
# ##bcftools_viewCommand=view -S 2.txt test_sample.vcf.gz; Date=Fri May 26 10:42:48 2023
# #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	PI423734	PI424241	PI437749	PI506711B	PI549076A	PI567453	PI594245B	PI184047	PI229339	PI398202
# Chr01	24952	ss715578788	A	G	.	.	.	GT	0/0	0/0	1/1	1/1	0/0	0/0	0/0	1/1	1/1	0/0
# ......

# After read the file, some transformation steps should be executed to fit the model input.
# In SoyDNGP, a 3*206*206 shaped matrix is needed as input. For detailed explanation, please read README.md.


class Reader:
    # initial class Reader, load the needed SNP file
    def __init__(self):
        self.columns = []
        self.indexes = None

    # read the file from vcf_path, return a transposed dataframe, whose columns are SNP IDs, and rows are samples' IDs
    def readVCF(self, vcf_path, reset=True):
        # open the .vcf file
        with open(vcf_path, "r") as f:
            # if the line does not start as "##", this line is the header line
            header = next((i for i, line in enumerate(f) if not line.startswith('##')), 0)
        # read the file from header row
        df = cudf.read_csv(vcf_path, header=header, sep='\t', dtype=str)
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
            df[column] = temp_series.astype(cp.int32)
            # delete useless variables
            del temp_series, mask
        # transpose the DataFrame, now the SNP IDs are columns, and sample IDs are indexes
        df = df.transpose()
        # keep the indexes and columns
        self.indexes = df.index.values_host
        self.columns = df.columns
        # drop indexes and reduce memory consumption
        if reset:
            df.reset_index(drop=True, inplace=True)
        return df.astype(cp.int32)


# convert origin matrix to one-hot matrix, and resize every samples
def one_hot(matrix):
    # repeat 3 times to 3D, in this way, transforming to one-hot matrix does not need explicit loop, which could perform better
    matrix_3D = cp.zeros((matrix.shape[0], matrix.shape[1], 3))
    # the rule for one-hot is like:
    #             '1/1' or '1|1'        '0/1' or '0|1'        other
    # channel 1          1                     1                0
    # channel 2          1                     0                1
    # channel 3          0                     1                1
    # for the first layer, only the first and second condition will be valued as 1
    matrix_3D[:, :, 0] = cp.where(cp.isin(matrix[:, :], cp.array([1, 2])), 1, 0)
    # for the second layer, only the second condition will be valued as 0
    matrix_3D[:, :, 1] = cp.where(cp.isin(matrix[:, :], cp.array(2)), 0, 1)
    # for the third layer, only the first condition will be valued as 0
    matrix_3D[:, :, 2] = cp.where(cp.isin(matrix[:, :], cp.array(1)), 0, 1)
    # resize every sample to shape of model input
    # set an index
    n = -1
    # create batches of resized matrix
    sample_resized = cp.empty((matrix.shape[0], 3, 206, 206)).astype(cp.float32)
    for sample in matrix_3D:
        # !!!!!(cite from numpy v1.25 document)
        # If the new array is larger than the original array, then the new array is filled with repeated copies of a.
        # Note that this behavior is different from a.resize(new_shape) which fills with zeros instead of repeated copies of a.
        # !!!!!
        sample = cp.resize(sample, (206, 206, 3))
        # transpose the order of sample's channels, for cp.reshape might change its order
        # for example, before reshape, sample has order [a, b, c], which might be changed after reshape, like [1, c, b, a]
        # to keep the result's order, here a transposition is significant
        sample = cp.transpose(sample, (2, 0, 1))
        # the first dimension is batch size, just for fitting the input shape
        sample = cp.reshape(sample, (1, 3, 206, 206))
        n += 1
        sample_resized[n, :, :, :] = sample
    return sample_resized

# an example
# r = Reader()  # instantiate class Reader
# print(len(r.SNPlist))  # get length of needed SNP list
# df = r.readVCF(r"D:\Projects\website\SoybeanNext\example\10_test_examples.vcf")  # get the processed dataframe
# df_filter, isMissing = r.SNPfilter(df)  # get the filtered dataframe
# sample_resized, indexlist = r.one_hot(df_filter)  # convert to one-hot matrix and resize every samples

