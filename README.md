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

## Training CenterNet

## Saving Detections

## Online Tube Generation

## Performance

###### ST action localization results (v-mAP) on UCF101-24
|  Method            |     f-mAP    |                    v-mAP                     ||
|                    |    @ 0.5     |   0.2    |    0.5   |   0.75   |   0.5:0.95   |  
| :----------------: |:------------:| :-------:| :-------:| :-------:| :----------: |


###### ST action localization results (v-mAP) on J-HMDB21
|  Method            |              |          |            |                |                |
| :----------------: |:------------:| :-------:| :--------: | :------------: | :------------: |

###### Inference Run Time Analysis
|  Framework Module  |              |          |            |                |                |                | 
| :----------------: |:------------:| :-------:| :--------: | :------------: | :------------: | :------------: |

## Citation

## Reference
