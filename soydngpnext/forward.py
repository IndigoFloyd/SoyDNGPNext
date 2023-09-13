import onnxruntime as onnx
import time
import cupy as np
import yaml
import pandas as pd
from soydngpnext.reader import Reader, one_hot
from soydngpnext.utils import *
import os
path = os.path.dirname(__file__)

# This module is used for predicting and analysis the results.
# Onnxruntime could run on GPU or CPU, which depends on your installed version of this package.
# If you would like to run it on your GPU, uninstall 'onnxruntime' at first, then install 'onnxruntime-gpu'.


class Forward:
    def __init__(self, traits, batch_size=1):
        # the path of models, it should be like:
        # ../
        # onnx/
        #   -- model1.onnx
        #   -- modeln.onnx
        self.modelsPath = fr'{path}/data/onnx/'
        self.batch_size = batch_size
        os.makedirs(self.modelsPath, exist_ok=True)
        # list of supported Quality Traits and corresponding index to value dictionary
        with open(f"{path}/data/p_trait.yaml") as p:
            self.data_p = yaml.safe_load(p)
        with open(f"{path}/data/n_trait.yaml") as n:
            self.data_n = yaml.safe_load(n)
        # traits to predict
        self.traits = list(set(traits))
        # preload needed models
        self.url = "http://xtlab.hzau.edu.cn/downloads/"
        for trait in self.traits:
            name = trait + '_best.onnx'
            # if model does not exist, download it
            if not os.path.isfile(f"{self.modelsPath + name}"):
                downloads(self.modelsPath, name)
            self.models = {trait: onnx.InferenceSession(self.modelsPath + name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])}

    # input is a list of samples
    # here, unpack the input data and split results by different trait types
    def forward(self, index_list, input_data=None):
        # transform the cupy array to numpy array (to avoid onnxruntime errors)
        input_data = np.asnumpy(input_data)
        # outputs of final results
        df = pd.DataFrame(None, columns=self.traits, index=index_list)
        # predict each trait for each sample
        for trait, model in self.models.items():
            # determine if this trait is a Quality Trait
            if trait in self.data_p.keys():
                # results
                results = []
                # indexed levels dictionary of this trait
                traitLevels = self.data_p[trait]
                # generate batched data and forward them
                for batch in self.batch_generator(input_data):
                    # raw results of this batch, its shape is batch * levels' amount
                    out = model.run(['output'], {'input': batch})[0]
                    # turn raw results to indexes
                    out = out.argmax(axis=1)
                    # merge output to list results, for matching levels in only one loop
                    results += out.tolist()
                df[trait] = [traitLevels[result] for result in results]
            # this trait is a Quantity Trait
            else:
                # results
                results = []
                traitMax = self.data_n[trait]['max']
                traitMin = self.data_n[trait]['min']
                traitDiff = traitMax - traitMin
                for batch in self.batch_generator(input_data):
                    # raw results of this batch, its shape is batch * levels' amount
                    out = model.run(['output'], {'input': batch})[0]
                    # merge output to list results, for matching levels in only one loop
                    results += out.tolist()
                # denormalize the result, use .get() method to transform cupy array to numpy array
                df[trait] = (np.array(results) * traitDiff + traitMin).get()
        return df

    # slice the input data into batches
    def batch_generator(self, input_data):
        start = 0
        # which batch is running now
        times = 1
        while start < input_data.shape[0]:
            end = start + self.batch_size
            # load input data batch by batch
            yield input_data[start:end, :, :, :]
            start = end
            times += 1

    # output DataFrame to .csv
    def output_dataframe(self, dataframe, output_path):
        dataframe.to_csv(f"{output_path}/result.csv")

    # the pipeline of whole execution, return result DataFrame
    def run(self, df_path, output=True):
        # instantiate class Reader
        r = Reader()
        s = time.time()
        # get the processed dataframe
        df = r.readVCF(rf"{df_path}")
        e = time.time()
        print(f"readVCF————{e - s:.2f}s")
        s = time.time()
        # convert to one-hot matrix and resize every samples
        index_list = r.indexes
        sample_resized = one_hot(df.values)
        e = time.time()
        print(f"one_hot————{e - s:.2f}s")
        s = time.time()
        # predict and get results
        output_path = outpath('predict')
        df = self.forward(index_list, sample_resized)
        e = time.time()
        print(f"forward————{e - s: .2f}s")
        if output:
            self.output_dataframe(df, output_path)
        return df

# an example
# f = Forward(['Hgt'], 10)
# f.run('data/10_test_examples.vcf')
