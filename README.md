# Koopy
Official implementation of Koopy.
Python version 3.13.
## Installation

Clone the repo:

```bash
git clone https://github.com/Koopy29542/Koopy.git
cd Koopy
pip install -r requirements.txt
```

The original datasets generated during our experimentations have been uploaded to here:[Drive](https://drive.google.com/file/d/1HvLqsX4YWHW1jplOqJwZSbFtcbqBBnhw/view?usp=sharing). Please replace the lobby2 and lobby3 folders within the koopy folder when utilizing these datasets.

We have placed the necessary data generation methods for the linear, gp, and Koopman methods within the koopy/conformal_prediction/generate_predictions folder. For easy use, please run the following code:
```
cd koopy
cd conformal_prediction
bash ./generate_all.sh
```
Comparison results utilized within the paper, including the relevant images, can be generated through the related Python files in the compare_predictions folder.
