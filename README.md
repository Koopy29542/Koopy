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

The original datasets generated during our experimentations have been uploaded to here:[Drive](https://drive.google.com/file/d/1HvLqsX4YWHW1jplOqJwZSbFtcbqBBnhw/view?usp=sharing) Please replace the lobby2 and lobby3 folder within the koopy folder when utilizing these datasets.

We have placed the neccesary data generation methods for the linear ,gp, and koopman methods within the koopy/conformal_prediction/generate_predictions folder. For easy use please run the following code:
```
cd koopy
cd conformal_prediction
bash ./generate_all.sh
```
Comparison results utililzed within the dataset ,including the relevant images , can be generated through the related python files in the compare_predictions folder.
