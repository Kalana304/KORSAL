# Key-point Detection based Online Real-Time Spatio-Temporal Action Localization
> Kalana Abeywardena, Sakuna Jayasundara, Sachira Karunasena, Shechem Sumanthiran, Dr. Peshala Jayasekara, Dr. Ranga Rodrigo

<p align='justify'>
Real-time and online action localization in a video is a critical yet highly challenging problem. Accurate action localization requires utilization of both temporal and spatial information. Recent attempts achieve this by using computationally intensive 3D CNN architectures or highly redundant two-stream architectures with optical flow, making them both unsuitable for real-time, online applications. To accomplish activity localization under highly challenging real-time constraints, we propose utilizing fast and efficient key-point based bounding box prediction to spatially localize actions. We then introduce a tube-linking algorithm that maintains the continuity of action tubes temporally in the presence of occlusions. Further, we eliminate the need for a two-stream architecture by combining temporal and spatial information into a cascaded input to a single network, allowing the network to learn from both types of information. Temporal information is efficiently extracted using a structural similarity index map as opposed to computationally intensive optical flow. Despite the simplicity of our approach, our lightweight end-to-end architecture achieves state-of-the-art frame-mAP on the challenging UCF101-24 dataset, demonstrating a performance gain of 6.4% over the previous best online methods. We also achieve state-of-the-art video-mAP results compared to both online and offline methods. Moreover, our model achieves a frame rate of 41.8 FPS, which is a 10.7% improvement over contemporary real-time methods.
</p>

<p align="center">
  <img src="figures/NewArchitecture.png">
  <em>Proposed Architecture</em>
</p>

## Highlights
- Utilize key-point-based detection architecture for the first time for the task of ST action localization, which reduces model complexity and inference time over traditional anchor-box-based approaches.
- Demonstrate that the explicit computation of OF is unnecessary and that the SSIM index map obtains sufficient inter-frame temporal information.
- A single network provided with both spatial and temporal information, and allowing it to extract necessary information through discriminative learning.
- An efficient tube-linking algorithm that extrapolates the tubes for a short period using past detections for real-time deployment.

## Table of Content
  1. [Installation](#installation)
  2. [Datasets](#datasets)
  3. [Training CenterNet](#training-centernet)
  3. [Saving Detections](#saving-detections)
  4. [Online Tube Generation](#online-tube-generation)
  5. [Performance](#performance)
  6. [Citation](#citation)
  7. [Reference](#reference)

## Installation

The code was tested on Ubuntu 18.04, with Anaconda Python 3.7 and PyTorch v1.4.0. NVIDIA GPUs are needed for both training and testing. After install Anaconda:

0. [Optional but recommended] create a new conda environment and activate the environment.
```
conda create --name CenterNet python=3.7
conda activate CenterNet
```
1. Install pytorch 1.4.0:
```
conda install pytorch=1.4.0 torchvision -c pytorch
```
Based on original [repository](https://github.com/xingyizhou/CenterNet), there can be slight reduction in performances for spatial localization with `cudann batch normalization` enabled. You can manually open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`. 

2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
```
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
make
python setup.py install --user
```

3. Install the requirements:
```
pip install -r requirements.txt
```


## Datasets
<p align='justify'>
We evaluate our framework on two datasets, <a href=https://www.crcv.ucf.edu/data/UCF101.php>UCF101-24</a> and <a href=http://jhmdb.is.tue.mpg.de/>J-HMDB21</a>. UCF101-24 is a subset of UCF101<sup>[1]</sup> dataset with ST labels, having 3207 untrimmed videos with 24 action classes, that may contain multiple instances for the same action class. J-HMDB-21 is a subset of the HMDB51<sup>[2]</sup> dataset having 928 temporally trimmed videos with 21 actions, each containing a single action instance. 

Download the datasets and extract the frames. Place the extracted frames in <emp>rgb-images</emp> in the respective dataset directory in [Datasets](https://github.com/Kalana304/KORSAL/tree/main/Datasets). The data directory should look as follows:
</p>

<p align="center">
  <img src="figures/sample directory tree.png">
  <em>Sample directory tree for J-HMDB21</em>
</p>

## Training CenterNet

### Setting up the CenterNet

When seting up CenterNet, we followed the instructions mentioned in their [official repository](https://github.com/xingyizhou/CenterNet). The modified scripts of CenterNet for Action Detection are provided in this [repository](https://github.com/Kalana304/KORSAL/tree/main/CenterNet). 

To install DCNv2, follow the below instructions:

1. **Build NMS**

```
cd CenterNet\src\lib\external
#python setup.py install
python setup.py build_ext --inplace
```

Comment out the parameter in setup.py when building `nms` extension to solve invalid numeric argument `/Wno-cpp`: `#extra_compile_args=["-Wno-cpp", "-Wno-unused-function"]` (the provided script by us has made the changes already).

2. **Clone and build original DCNv2**

```
cd CenterNet\src\lib\models\networks
rm -rf DCNv2
git clone https://github.com/CharlesShang/DCNv2
```

After cloning the original DCNv2, navigate to the directory and make the following changes:

```
cd DCNv2
vim cuda/dcn_va_cuda.cu
"""
# extern THCState *state;
THCState *state = at::globalContext().lazyInitCUDA();
"""
```

Finally, execute the following command to build DCNv2:

```
python setup.py build develop
```

### Training from the scratch
To train from the scratch with either UCF101-24 or J-HMDB21 datasets, the following command can be run. 

```
python CUDA_VISIBLE_DEVICES=0 python main_SMDouble.py --dataset <dataset> --gpus <gpu id> --exp_id <save dir name> --task doubleSM --num_epochs <epochs (default: 60)> --variant <variation (default: 1)> 
```

### Resuming from saved checkpoint
To resume the training from the last checkpoint saved, run the following command.

```
python CUDA_VISIBLE_DEVICES=0 python main_SMDouble.py --dataset <dataset> --gpus <gpu id> --exp_id <save dir name> --task doubleSM --num_epochs <epochs (default: 60)> --variant <variation (default: 1)> --resume 
```

Further, to resume the training from a specific chekpoint saved, run the following command.

```
python CUDA_VISIBLE_DEVICES=0 python main_SMDouble.py --dataset <dataset> --gpus <gpu id> --exp_id <save dir name> --task doubleSM --num_epochs <epochs (default: 60)> --variant <variation (default: 1)> --resume --load_model <path to the saved model>
```

### Transfer Learning using the best checkpoint
To tranfer learn from a pre-trained checkpoint, run the following command.

```
python CUDA_VISIBLE_DEVICES=0 python main_SMDouble.py --dataset <dataset> --gpus <gpu id> --exp_id <save dir name> --task doubleSM --num_epochs <epochs (default: 60)> --variant <variation (default: 1)> --load_model /path/to/checkpoint
```

The pre-trained model checkpoints trained on J-HMDB21 and UCF101-24 datasets can be downloaded from [here](https://drive.google.com/drive/folders/1jb5QfujoQngP4QqyN-PGvYba1jwhb9th?usp=sharing). Place the chekpoints at `./CenterNet/exp/$DATASET_NAME/dla34/rgb/$CHKPT_NAME` to be compatible with the directory path definitions in the Centernet scripts. 

## Saving Detections
For evaluation, the spatial detections needs to be saved as <I>.mat </I> files. First, navigate to ./Save Detections/ and execute the following command:
```
python CUDA_VISIBLE_DEVICES=0 python SaveDetections.py --dataset <dataset> --ngpu <gpu id> --exp_id <save dir name> --task doubleSM --frame_gap <default:1> --variant <default:1> --load_model /path/to/checkpoint --result_root /path/to/detections
```

## Online Tube Generation

After the spatial detections are saved for each video, the action tubes and paths are generated using the proposed online tube generation algorithm and its variation that are based on the [original implementation](https://github.com/Kalana304/realtime-action-detection) which is also provided for comparison. The codes can be found in `./online-tubes/`.
 - To run the code, you will need to install MATLAB. You can install a [free trial](https://www.mathworks.com/products/matlab.html) for testing purposes. Make sure you add the MATLAB installation path to the conda environment if you are executing scripts using command line.
 - If you only have command line priviledges, you can install [Octave](https://wiki.octave.org/Octave_for_Debian_systems) and execute the tube generation.

### Executing with MATLAB

1. Navigate to the respective directory: 
```
cd ./online-tubes/EXP_without_BBOX
```
2. Change the paths based on where the data (saved detections) is located and results need saving in `I01onlineTubes.m` and `utils/initDatasetOpts.m`.
3. Execute `I01onlineTubes.m`. When executing using command line:
```
matlab -batch "I01onlineTubes.m"
```

### Executing with Octave

1. Navigate to the respective directory: 
```
cd ./online-tubes/EXP_without_BBOX
```
2. Change the paths based on where the data (saved detections) is located and results need saving in `I01onlineTubes.m` and `utils/initDatasetOpts.m`.
3. Execute `I01onlineTubes.m`. When executing using command line:
```
octave I01onlineTubes.m
```
*There can be errors when running the current scripts in Octave. This is due to `-v7.3` argument used in `save()` function in MATLAB scripts. You can simply remove the `-v7.3` argument in `save()` functions and run without errors.* 

## Performance
<p align='justify'>
We describe our experimental results and compare them with state-of-the-art offline and online methods that use either RGB or both RGB and OF inputs. Further, for comparison
we present results on action localization using only the appearance (A) information extracted by a single frame. The results of our proposed method presented in Table demonstrate that we are able to achieve state-of-the-art performance.
</p >

###### ST action localization results (v-mAP) on UCF101-24 and J-HMDB21 datasets
<table  width="100%">
  <col>
  <colgroup span="5"></colgroup>
  <colgroup span="4"></colgroup>
  <tr>
    <th rowspan="3">Method</th>
    <th colspan="5" scope="colgroup">UCF101-24</th>
    <th colspan="5" scope="colgroup">J-HMDB21</th>
    <th rowspan="3">FPS</th>
  <tr>
    <th rowspan="2">f-mAP <br/> @0.5</th>
    <th colspan="4" scope="colgroup">v-mAP</th>
    <th rowspan="2">f-mAP <br/> @0.5</th>
    <th colspan="4" scope="colgroup">v-mAP</th>
  </tr>
  <tr>
    <th scope="col">0.2</th> <th scope="col">0.5</th> <th scope="col">0.75</th> <th scope="col">0.5:0.95</th>
    <th scope="col">0.2</th> <th scope="col">0.5</th> <th scope="col">0.75</th> <th scope="col">0.5:0.95</th>
  </tr>
  <tr>
    <td scope="row">Saha et al.<sup>[3]</sup></td>
    <td align="center" valign="center">-</td> <td align="center" valign="center">66.6</td> <td align="center" valign="center">36.4</td> <td align="center" valign="center">7.9</td> <td align="center" valign="center">14.4</td>
    <td align="center" valign="center">-</td> <td align="center" valign="center">72.6</td> <td align="center" valign="center">71.5</td> <td align="center" valign="center">43.3</td> <td align="center" valign="center">40.0</td> <td align="center" valign="center">4</td>
  </tr>
  <tr>
    <td scope="row">Peng et al.<sup>[4]</sup></td>
    <td align="center" valign="center">65.7</td> <td align="center" valign="center">72.9</td> <td align="center" valign="center">-</td> <td align="center" valign="center">-</td> <td align="center" valign="center">-</td>
    <td align="center" valign="center">58.5</td> <td align="center" valign="center">74.3</td> <td align="center" valign="center">73.1</td> <td align="center" valign="center">-</td> <td align="center" valign="center">-</td> <td align="center" valign="center">-</td>
  </tr>
  <tr>  
    <td scope="row">Zhang et al.<sup>[5]</sup></td>
    <td align="center" valign="center">67.7</td> <td align="center" valign="center">74.8</td> <td align="center" valign="center">46.6</td> <td align="center" valign="center">16.7</td> <td align="center" valign="center">21.9</td>
    <td align="center" valign="center">37.4</td> <td align="center" valign="center">-</td> <td align="center" valign="center">-</td> <td align="center" valign="center">-</td> <td align="center" valign="center">-</td> <td align="center" valign="center">37.8</td>
  </tr>
  <tr>
    <td scope="row">ROAD+AF<sup>[6]</sup></td>
    <td align="center" valign="center">-</td> <td align="center" valign="center">73.5</td> <td align="center" valign="center">46.3</td> <td align="center" valign="center">15.0</td> <td align="center" valign="center">20.4</td>
    <td align="center" valign="center">-</td> <td align="center" valign="center">70.8</td> <td align="center" valign="center">70.1</td> <td align="center" valign="center">43.7</td> <td align="center" valign="center">39.7</td> <td align="center" valign="center">7</td>
  </tr>
  <tr>
    <td scope="row">ROAD+RTF<sup>[6]</sup></td>
    <td align="center" valign="center">-</td> <td align="center" valign="center">70.2</td> <td align="center" valign="center">43.0</td> <td align="center" valign="center">14.5</td> <td align="center" valign="center">19.2</td>
    <td align="center" valign="center">-</td> <td align="center" valign="center">66.0</td> <td align="center" valign="center">63.9</td> <td align="center" valign="center">35.1</td> <td align="center" valign="center">34.4</td> <td align="center" valign="center">28</td>
  </tr>
  <tr>
    <td scope="row">ROAD (A)<sup>[6]</sup></td>
    <td align="center" valign="center">-</td> <td align="center" valign="center">69.8</td> <td align="center" valign="center">40.9</td> <td align="center" valign="center">15.5</td> <td align="center" valign="center">18.7</td>
    <td align="center" valign="center">-</td> <td align="center" valign="center">60.8</td> <td align="center" valign="center">59.7</td> <td align="center" valign="center">37.5</td> <td align="center" valign="center">33.9</td> <td align="center" valign="center">40</td>
  </tr> 
  <tr>
    <td scope="row"> <strong>Ours (A)</strong> </td>
    <td align="center" valign="center">71.8</td> <td align="center" valign="center">70.2</td> <td align="center" valign="center"><strong>44.3</strong></td> <td align="center" valign="center">16.6</td> <td align="center" valign="center"><strong>20.6</strong></td>
    <td align="center" valign="center"><strong>51.2</strong></td> <td align="center" valign="center">59.3</td> <td align="center" valign="center">59.2</td> <td align="center" valign="center">48.2</td> <td align="center" valign="center"><strong>41.2</strong></td> <td align="center" valign="center"><strong>52.9</strong></td>
  </tr>
  <tr>
    <td scope="row"><strong>Ours</strong> </td>
    <td align="center" valign="center"><strong>74.7</strong></td> <td align="center" valign="center"><strong>72.7</strong></td> <td align="center" valign="center"><strong>43.1</strong></td> <td align="center" valign="center"><strong>16.8</strong></td> <td align="center" valign="center"><strong>20.2</strong></td>
    <td align="center" valign="center">50.5</td> <td align="center" valign="center">58.9</td> <td align="center" valign="center">58.4</td> <td align="center" valign="center"><strong>49.5</strong></td> <td align="center" valign="center">40.6</td> <td align="center" valign="center"><strong>41.8</strong></td>
  </tr>
</table>

<p align='justify'>
We analyze the inference times for different variations of our pipeline based on the different modules in the framework and the overall inference time in the below Table.
Evidently, any preprocessing will have an impact on the inference time. Thus, the SS-map achieves a balance between the run-time and the accuracy over the other variations in the framework.
</p>

###### Inference Run Time Analysis
|  Framework Module  |    Ours   |   A + DSIM |   A + I<sub>t-1</sub> |     A      |     A + RTF    |      A + AF    | 
| :---------------- |:---------:| :---------:| :-----------: | :-----------: | :------------: | :------------: |
|Temporal INFO EXT (ms)|  5.0  | 5.0 |  -  | -  |  7.0  | 110.0 |
|Detection network (ms)| 16.4 | 16.4 | 16.4 | 16.4 | 16.4 | 16.4 |
|Tube generation time (ms)| 2.5 | 2.5 | 2.5 | 2.5 | 3.0 | 3.0 |
|<strong>Overall (ms)</strong>| 23.9 | 23.9 | 18.9 | 18.9 | 26.4 | 129.4 |


## Citation
```
@article{abeywardena2021korsal,
  title={KORSAL: Key-point Detection based Online Real-Time Spatio-Temporal Action Localization},
  author={Abeywardena, Kalana and Sumanthiran, Shechem and Jayasundara, Sakuna and Karunasena, Sachira and Rodrigo, Ranga and Jayasekara, Peshala},
  journal={arXiv preprint arXiv:2111.03319},
  year={2021}
}
```

```
@INPROCEEDINGS{10288973,
  author={Abeywardena, Kalana and Sumanthiran, Shechem and Jayasundara, Sakuna and Karunasena, Sachira and Rodrigo, Ranga and Jayasekara, Peshala},
  booktitle={2023 IEEE Canadian Conference on Electrical and Computer Engineering (CCECE)}, 
  title={KORSAL: Key-Point Based Online Real-Time Spatio-Temporal Action Localization}, 
  year={2023},
  volume={},
  number={},
  pages={279-284},
  doi={10.1109/CCECE58730.2023.10288973}}
```

## Reference
[1] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. \\

[2] H. Jhuang, J. Gall, S. Zuffi, C. Schmid, and M. J. Black. Towards understanding action recognition. In International Conf. on Computer Vision (ICCV), pages 3192–3199,
December 2013.

[3] Suman Saha, Gurkirt Singh, Michael Sapienza, Philip HS Torr, and Fabio Cuzzolin. Deep learning for detecting multiple space-time action tubes in videos. arXiv preprint
arXiv:1608.01529, 2016.

[4] Xiaojiang Peng and Cordelia Schmid. Multi-region two-stream r-cnn for action detection. In ECCV, pages 744–759. Springer, 2016.

[5] Dejun Zhang, Linchao He, Zhigang Tu, Shifu Zhang, Fei Han, and Boxiong Yang. Learning motion representation for real-time spatio-temporal action localization. Pattern
Recognition, 103:107312, 2020.
