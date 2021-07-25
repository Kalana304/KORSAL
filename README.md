# Key-point Detection based Online Real-Time Spatio-Temporal Action Localization
> Kalana Abeywardena, Sakuna Jayasundara, Sachira Karunasena, Shechem Sumanthiran, Dr. Peshala Jayasekara, Dr. Ranga Rodrigo

<p align='justify'>
Real-time and online action localization in a video is a critical yet highly challenging problem. Accurate action localization requires utilization of both temporal and spatial information. Recent attempts achieve this by using computationally intensive 3D CNN architectures or highly redundant two-stream architectures with optical flow, making them both unsuitable for real-time, online applications. To accomplish activity localization under highly challenging real-time constraints, we propose utilizing fast and efficient key-point based bounding box prediction to spatially localize actions. We then introduce a tube-linking algorithm that maintains the continuity of action tubes temporally in the presence of occlusions. Further, we eliminate the need for a two-stream architecture by combining temporal and spatial information into a cascaded input to a single network, allowing the network to learn from both types of information. Temporal information is efficiently extracted using a structural similarity index map as opposed to computationally intensive optical flow. Despite the simplicity of our approach, our lightweight end-to-end architecture achieves state-of-the-art frame-mAP of 74.7% on the challenging UCF101-24 dataset, demonstrating a performance gain of 6.4% over the previous best online methods. We also achieve state-of-the-art video-mAP results compared to
both online and offline methods. Moreover, our model achieves a frame rate of 41.8 FPS, which is a 10.7% improvement over contemporary real-time methods.
</p>

<p align="center">
  <img src="figures/NewArchitecture.png">
  <em>Proposed Architecture</em>
</p>

## Table of Content
  1. [Installation](#installation)
  2. [Datasets](#datasets)
  3. [Training CenterNet](#centernet)
  3. [Saving Detections](#detections)
  4. [Online Tube Generation](#tubegeneration)
  5. [Performance](#performance)
  6. [Citation](#citation)
  7. [Reference](#reference)

## Installation

## Datasets
We evaluate our framework on two datasets, [UCF101-24](https://www.crcv.ucf.edu/data/UCF101.php) and [J-HMDB-21](http://jhmdb.is.tue.mpg.de/). UCF101-24 is a subset of UCF101 [1] with ST labels, having 3207 untrimmed videos with 24 action classes, that may contain multiple instances for the same action class. J-HMDB-21 is a subset of the HMDB-51 dataset [2] having 928 temporally trimmed videos with 21 actions, each containing a single action instance. 

Download the datasets and extract the frames. Place the extracted frames in <emp>rgb-images</emp> in the respective dataset directory in [Datasets](https://github.com/Kalana304/KORSAL/tree/main/Datasets). The data directory should look as follows:

<p align="center">
  <img src="figures/sample directory tree.png">
  <em>Sample directory tree for JHMDB21</em>
</p>

## Training CenterNet

## Saving Detections

## Online Tube Generation

## Performance

###### ST action localization results (v-mAP) on UCF101-24
<table>
  <col>
  <colgroup span="4"></colgroup>
  <tr>
    <td rowspan="3">Method</td>
    <td rowspan="5">UCF101-24</td>
    <td rowspan="5">J-HMDB21</td>
  <tr>
    <td rowspan="2">f-mAP @0.5</td>
    <th colspan="4" scope="colgroup">v-mAP</th>
    <td rowspan="2">f-mAP @0.5</td>
    <th colspan="4" scope="colgroup">v-mAP</th>
  </tr>
  <tr>
    <th scope="col">0.2</th>
    <th scope="col">0.5</th>
    <th scope="col">0.75</th>
    <th scope="col">0.5:0.95</th>
  </tr>
  <tr>
    <th scope="row">Saha et al. [3]</th>
    <td>-</td> <td>66.6</td> <td>36.4</td> <td>7.9</td> <td>14.4</td>
  </tr>
  <tr>
    <th scope="row">Peng et al. [4]</th>
    <td>65.7</td> <td>72.9</td> <td>-</td> <td>-</td> <td>-</td>
  </tr>
  <tr>  
    <th scope="row">Zhang et al. [5]</th>
    <td>67.7</td> <td>74.8</td> <td>46.6</td> <td>16.7</td> <td>21.9</td>
  </tr>
  <tr>
    <th scope="row">ROAD+AF [6]</th>
    <td>-</td> <td>73.5</td> <td>46.3</td> <td>15.0</td> <td>20.4</td>
  </tr>
  <tr>
    <th scope="row">ROAD+RTF [6]</th>
    <td>-</td> <td>70.2</td> <td>43.0</td> <td>14.5</td> <td>19.2</td>
  </tr>
  <tr>
    <th scope="row">ROAD (A) [6]</th>
    <td>-</td> <td>69.8</td> <td>40.9</td> <td>15.5</td> <td>18.7</td>
  </tr>
  <tr>
    <th scope="row">Ours (A) [6]</th>
    <td>71.8</td> <td>70.2</td> <td>44.3</td> <td>16.6</td> <td>20.6</td>
  </tr>
  <tr>
    <th scope="row">Ours [6]</th>
    <td>74.7</td> <td>72.7</td> <td>43.1</td> <td>16.8</td> <td>20.2</td>
  </tr>
</table>

###### ST action localization results (v-mAP) on J-HMDB21
<table>
  <col>
  <colgroup span="4"></colgroup>
  
  <tr>
    <td rowspan="2">Method</td>
    <td rowspan="2">f-mAP @0.5</td>
    <th colspan="4" scope="colgroup">v-mAP</th>
  </tr>
  <tr>
    <th scope="col">0.2</th>
    <th scope="col">0.5</th>
    <th scope="col">0.75</th>
    <th scope="col">0.5:0.95</th>
  </tr>
  <tr>
    <th scope="row">Teddy Bears</th>
    <td>50,000</td>
    <td>30,000</td>
    <td>100,000</td>
    <td>80,000</td>
    <td>80,000</td>
  </tr>
</table>

###### Inference Run Time Analysis
|  Framework Module  |    Ours   |   A + DSIM |   A + I<sub>t-1</sub> |     A      |     A + RTF    |      A + AF    | 
| :---------------- |:---------:| :---------:| :-----------: | :-----------: | :------------: | :------------: |
|Temporal INFO EXT (ms)|  5.0  | 5.0 |  -  | -  |  7.0  | 110.0 |
|Detection network (ms)| 16.4 | 16.4 | 16.4 | 16.4 | 16.4 | 16.4 |
|Tube generation time (ms)| 2.5 | 2.5 | 2.5 | 2.5 | 3.0 | 3.0 |
|<strong>Overall (ms)</strong>| 23.9 | 23.9 | 18.9 | 18.9 | 26.4 | 129.4 |


## Citation

## Reference
[1] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. \\

[2] H. Jhuang, J. Gall, S. Zuffi, C. Schmid, and M. J. Black. Towards understanding action
recognition. In International Conf. on Computer Vision (ICCV), pages 3192–3199,
December 2013.
