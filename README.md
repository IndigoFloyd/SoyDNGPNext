# SoyDNGPNext Documentation V0.0.1

SoyDNGPNext is a deep learning driving bioinformatical toolkit, which performs good on soybean datasets, and is permitted to apply on other organisms. This documentation could also be treated as a tutorial.

## Structure

SoyDNGPNext adopts the flat structure, and the basic structure is as follows.

```
.
│  data_process.py  
│  eval.py           
│  forward.py
│  reader.py
│  reader_cpu.py
│  remodel.py
│  SoyDNGP.py
│  train.py
│  tree.txt
│  utils.py
│  weight_map.py
│  __init__.py
│  
├─data  # data and config files
│      model.yaml  # contains the model structure
│      n_trait.yaml  # contains Quantity Traits and max and minimum values
│      p_trait.yaml  # contains Quality Traits and levels of each trait
│      traits.yaml  # users should write Quantity and Quality traits before training
│      
└─runs  # store train or predict results
```





## python APIs

### SoyDNGPNext.utils

In this module, kinds of utility functions are included.

#### utils.downloads(path, name)

Download pointed-```name``` file to ```path``` from the URL http://xtlab.hzau.edu.cn/downloads/name.

For example:

```python
import SoyDNGPNext as sn
# download the minist .vcf file example
sn.utils.downloads("examples/", "10_test_examples.vcf")  
# download from "http://xtlab.hzau.edu.cn/downloads/10_test_examples.vcf" to SoybeanNext/examples/
```

The ```path``` is related to the package path. If you setup the package in path ```anaconda3/envs/test/lib/python3.10/site-packages/```, file will be downloaded to ```anaconda3/envs/test/lib/python3.10/site-packages/SoybeanNext/path```.

Sometimes the file might be broken, when something goes wrong, remember check whether the file is complete or not.

#### utils.exportAsONNX(inputPath, input_W=206, input_H=206, input_channel=3, cuda=True)

Export .pt weights to .onnx models. Faster forwarding and broader application are the reasons why we choose ONNX as our standard model format.

For example:

```python
import SoyDNGPNext as sn
# export all .pt weights in path
sn.utils.exportAsONNX("path_to_pt", input_W=206, input_H=206, input_channel=3, cuda=True)
# the input feature maps are in shape [3, 206, 206]
# models and dummy tensors will be loaded on CUDA device
```

This function is based on ```torch.onnx.export```, the output .onnx models support dynamic input batch size. So when input some data later, the data should be shaped as $[batch_size, 3, 206, 206]$ at first.

#### utils.outpath(work)

Create directory named by parameter ```work```.  The whole path is:

```python
import os
path = os.path.dirname(__file__)  # where the script runs
new_path = f'{path}/runs/{work}/{work}{number}'  # the returned outpath
```

After running this method time by time, the folder might be like:

```
.      
├─predict
│  ├─predict1
│  └─predict2
└─train
    ├─train1
    └─train2
```



### SoyDNGPNext.Reader

The ```Reader``` class is defined in ```reader.py```. In this module, .vcf files will be loaded and processed in high efficiency with ```cudf``` and ```cupy```.

#### Reader().readVCF(vcf_path, reset=True)

Read .vcf file from ```vcf_path``` as useful DataFrame type data.

To make good use of ```cudf``` and ```cupy```, values will be replaced as:

| original value  | replaced value |
| --------------- | -------------- |
| '1/1' or '1\|1' | '1'            |
| '0/1' or '0\|1' | '2'            |
| other values    | '3'            |

Then, the datatype of the DataFrame will be set to int32, and the DataFrame will be transposed. Indexes and columns are saved in Intra-class variables ```self.indexes``` and ```self.columns```. To reduce consumption of memory, indexes will be dropped by default.

For example:

```python
from SoybeanNext.reader import Reader
r = Reader()
# get the processed dataframe
df = r.readVCF(rf"{df_path}")
print(r.indexes, r.columns, sep='\n')
```

### SoyDNGPNext.one_hot(matrix)

One-hot the genotype data matrix, and reshaped them into $[total_batchsize, 3, 206, 206]$ size.

Benefited from ```cupy``` and ```Reader().readVCF```, this method runs efficiently. The one-hot rules are as follow:

|          | 1    | 2    | 3    |
| -------- | ---- | ---- | ---- |
| channel1 | 1    | 1    | 0    |
| channel2 | 1    | 0    | 1    |
| channel3 | 0    | 1    | 1    |

For example:

```python
from SoybeanNext.reader import one_hot
sample_resized = one_hot(df.values)
```

### SoyDNGPNext.Reader_CPU and SoyDNGPNext.one_hot_CPU

Using the ```cupy``` and ```cudf``` libraries to accelerate will cause a lot of graphics card usage, so we also provide their CPU implementation. During the prediction process, we default to the GPU-accelerated implement, and during the training process, we default to the original CPU implement.

They are the same to use. However, **the ```reset``` switch is False by default in SoyDNGPNext.ReaderCPU**. 

### SoyDNGPNext.Forward

This module is used for predicting from one-hot genotype data.

Make sure your models are put in ```data/onnx``` folder, and ```data/p_trait.yaml``` and ```data/n_trait.yaml``` are correctly written.

```yaml
# an example of p_trait.yaml
# name of the quality trait
MG:
# level indexes and values
  0: VI
  1: IV
  2: III
  3: X
  4: O
  5: IX
  6: II
  7: I
  8: V
  9: VIII
  10: VII
```

```yaml
# an example of n_trait.yaml
# name of the quantity trait
protein:
# max and minimum values, for normalizing and denormalizing
  max: 57.9
  min: 31.7
```

If predicted traits are in the default .yaml configs, the .onnx models will be downloaded from our server if they are not there.

To initialize the ```Forward``` class, for example:

```python
import SoyDNGPNext as sn
# input traits list you want to predict, and set the batchsize (1 as default)
f = sn.Forward(['MG', 'protein'], 10)
```



#### Forward().forward(self, index_list, input_data=None):

Predict and do some after-processes. A DataFrame will be returned.

This method is gathered in ```Forward().run()```. 

#### Forward().batch_generator(input_data)

Slice the input data into batches.

This method is gathered in ```Forward().forward()```. 

#### Forward().output_csv()

Output DataFrame to .csv.

This method is gathered in ```Forward().run()```. 

#### Forward().run(self, df_path, output=True)

The pipeline of whole execution, return result DataFrame. 

If the switch ```output``` is True, then ```Forward().output_dataframe()``` will be called. The output .csv file will be saved in ```runs/predict/predict...```.

For example:

```python
f.run('data/10_test_examples.vcf')
```

### SoyDNGPNext.remodel(path, num_classes, show_structure=True)

This module is used for training your own model, you can reconstruct SoyDNGP to make sure the net is more adaptable to your dataset. You can build the model by revising ***model.yaml*** whose path is related to ```path```, if you want to show your model structure please set ```show_structure=True```. ```Num_classes``` is the number of categories for the classification task, if it is a regression task, please set ```num_classes=1```.

For example:

```python
from SoyDNGPNext.remodel import remodel
path = 'model.yaml'
num_class = 1
net = remodel(path,num_class,show_structure=True)
```

#### model.yaml

Model structure is defined in this file. You can design your CNN model by using the ***block*** with the specified format:  ***block name.str: (parameter list)***.

##### Block

* CNN_Block:(input_channel,out_channel,kernel_size,padding_size,stride,dropout_rate)
  * Include:
    - Convolutional layer
    - Batch normalization layer
    - Dropout layer  ( Configure the dropout rate by setting ```dropout_rate ``` )
* ReLU_:()
  - ReLU layer
* Linear_:(input_lenght,num_class,dropout_rate)
  * Include:
    - Flatten layer
    - Dropout layer
    - ReLU layer
    - Linear layer
* SE_attention:(input_channel, reduction)
  - Squeeze-and-Excitation attention
* CBAM_attention:(input_channel, reduction)
  - Convolutional Block Attention Module
* CA_attention:(input_channel,height,width,reduction)
  - Coordinate Attention
* Rediual_Block:(in_channel,out_channel,kernel_size,padding,stride,drop_out)
  * When the stride = 1 Rediual_Block is equal two CNN_Block which include:
    - CNN_Block1:(input_channel,out_channel,kernel_size,padding_size,1,dropout_rate)
    - CNN_Block2:(out_channel,out_channel,3,1,1,dropout_rate)

For example:

```yaml
model:
 SE_attention.1: (3,16)
 CNN_Block.1: (3,32,3,1,1,0.3)
 ReLU_.1: ()
 CNN_Block.2: (32,64,4,1,2,0.3)
 ReLU_.2: ()
 SE_attention.2: (64,16)
 Linear_.1: (1024,1,0.3)
```

### SoyDNGPNext.Train

This module is used for training with SoyDNGP default  model or custom models on your own datasets.

**If you have changed model structure in ```data/model.yaml```, it will be applied automatically in training process.**

#### How to prepare your own datasets

- Prepare the ```traits.csv```, ```genotype.vcf``` and ```traits.yaml```. They should be gathered in ```data``` directory.

  For example, if you want to train traits 'protein' and 'SCN3', you should write like:

  **traits.yaml**

  ```yaml
  # Quantity Traits
  n:
    [protein]
  # Quality Traits
  p:
    [SCN3]
  ```

  **traits.csv**

  | acid      | SCN3 | protein |
  | --------- | ---- | ------- |
  | PI219698  | S    | 41.3    |
  | PI253651A | S    | 42.6    |
  | PI347550A | S    | 44.7    |
  | ...       | ...  | ...     |

  **genotype.vcf**
  
  | #CHROM | POS   | ID          | REF  | ALT  | QUAL | FILTER | INFO             | FORMAT | PI594433A | ...  |
  | :----- | ----- | ----------- | ---- | ---- | ---- | ------ | ---------------- | ------ | --------- | ---- |
  | Chr01  | 24952 | ss715578788 | A    | G    | .    | PASS   | AC=9353;AN=36526 | GT     | 1\|1      | ...  |

Initialize the class ```train```, for example:

```python
from SoyDNGPNext import Train
# an example
t = Train("data/genotype.vcf", "data/traits.csv")
```

#### Train().train_n(percentage=0.05, epoch=5, weight_decay=1e-5, draw=True)

Train quantity traits model. By calling ```SoyDNGPNext.weight_decoder```, we could find out the first ```percentage``` samples which contribute most. Then, training ```epoch``` epochs, and set ```weight_decay``` as the parameter of ```torch.optim.Adam()```. Finally, if ```draw``` is True, the evaluation pictures will be drawn and saved in path such as ```runs/train/train1```.

By computing the correlation coefficient value, only the best model will be saved in the end. A dictionary will be returned, too.

```python
eval_dict = {'train_loss': [], 'test_loss': [], 'mse': [], 'r': [], 'trait': ''}
```



For example:

```python
# train 100 epochs
t.train_n(epoch=100)
```

#### Train().train_p(percentage=0.05, epoch=5, weight_decay=1e-5, draw=True)

Train quality traits model. By calling ```SoyDNGPNext.weight_decoder```, we could find out the first ```percentage``` samples which contribute most. Then, training ```epoch``` epochs, and set ```weight_decay``` as the parameter of ```torch.optim.Adam()```. Finally, if ```draw``` is True, the evaluation pictures will be drawn and saved in path such as ```runs/train/train1```.

For example:

```python
# train 100 epochs
t.train_p(epoch=100)
```

By computing the correlation coefficient value, only the best model will be saved in the end. A dictionary will be returned, too.

```python
eval_dict = {'train_loss': [], 'test_loss': [], 'acc': [], 'recall': [], 'precision': [], 'f1_score': [], 'trait': '', 'confusion_matrix': nd.array}
```

