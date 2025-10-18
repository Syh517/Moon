# Moon

This is code for the paper: [Moon: A Modality Conversion-based Efficient Multivariate Time Series Anomaly Detection (TKDE 2025) ](https://arxiv.org/abs/2510.01970)

Moon adopts the **MV-MTF** technique, which converts numerical multivariate time series into images, effectively capturing temporal information and providing richer modality representations for subsequent detection. The **Multimodal-CNN** model focuses on critical features and fuses multi-scale information from both image data and the original numerical data, thereby improving the accuracy and efficiency of anomaly detection. **SHAP-based anomaly explanation module** leverages SHAP values to quantify the contribution of each variable to anomalies, significantly enhancing the interpretability of detection results.

## Get started
### Get Data
 Put SMD (Server Machine Dataset) in folder ***./ServerMachineDataset***. SMD can be downloaded from [here](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset).


### Install dependencies (with python 3.8)
```bash
pip install -r requirements.txt
```

### Run Moon

To converts numerical multivariate time series into images, run *MTF_picture.py*
```bash
python MTF_picture.py
```

To train the abnormaly detection model, run *detection_model.py*
```bash
python detection_model.py
```

To detect anomalies and get the explanation, run *detect.py*
```bash
python detect.py
```