## Setup

Install the dependencies listed by running:
```
pip install -r requirement.txt
```
We use the dataset from [NSynth](https://drive.google.com/drive/folders/1YSuQkRdVYyZ7Dlk7Rh5QoAQOkE1IFC9U?usp=sharing).
Place the NSynth dataset under `./dataset/` with these directories:

```text
dataset/
├── nsynth-train/audio/
├── nsynth-train/examples.json
├── nsynth-valid/audio/
└── nsynth-valid/examples.json
```
Please note that the file format should be json/wav.

Run all commands from the repository root.

## Preprocessing

Extract Mel spectrograms and MFCC features for the training and validation sets. This creates the `.npy` files used by both training pipelines:

```bash
python code/preprocessing.py
```

## Machine Learning Method: k-NN classifier
### Training

Train and validate the k-NN classifier. The script searches over `k = 5, 10, ..., 50` and saves the best model to `./model/knn_model.pkl`:

```bash
python code/task_2_train.py
```

### Inference
The inference scripts expect the test split at `./dataset/nsynth-test/`, containing `audio/` and `examples.json`.
Run inference with the trained `./model/knn_model.pkl` model:

```bash
python code/task_2_inference.py
```

To use a different test-data directory:

```bash
python code/task_2_inference.py --data_path ./dataset/nsynth-test
```

The script prints top-1 and top-3 accuracy and saves the confusion matrix to
`./result_image/confusion_matrix_task_2.pdf`.

## Deep Learning Method: `MusicNet`
### Training

Train the convolutional neural network with the default settings (200 epochs, batch size 16, learning rate 0.001, and early stopping after 50 epochs without validation improvement):

```bash
python code/task_3_train.py
```

The training script uses log-scaled Mel spectrograms by default and saves the best checkpoint to `./checkpoint/model_best_log.pth`. To train without log-scaling, use:

```bash
python code/task_3_train.py --log_scale 0
```

Additional options can be changed from the command line, for example:

```bash
python code/task_3_train.py \
  --n_epochs 300 \
  --batch_size 32 \
  --lr 0.0005 \
  --early_stop 75 \
  --folder_path ./dataset
```

### Inference

Run inference with the log-scale checkpoint. The checkpoint must be available at `./model/model_best_log.pth`:

```bash
python code/task_3_inference.py \
  --data_path ./dataset/nsynth-test \
  --log_scale 1
```

For a checkpoint trained without log scaling, use `./model/model_best.pth` and run:

```bash
python code/task_3_inference.py \
  --data_path ./dataset/nsynth-test \
  --log_scale 0
```

The script prints top-1 and top-3 accuracy and saves the confusion matrix to `./result_image/confusion_matrix_task_3_log.pdf` or `./result_image/confusion_matrix_task_3.pdf`, respectively.
