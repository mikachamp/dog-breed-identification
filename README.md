# Dog Breed Identification

**A transfer-learning classifier that names a dog's breed from a photo. 78.1% validation accuracy across 120 breeds.**

I built it in TensorFlow on Google Colab. It started as coursework in the Zero To Mastery Machine Learning & Data Science course, and then I kept going with my own experiments. The most useful result was a failure: fine-tuning the pre-trained model made it much worse, and leaving it frozen won by a wide margin.

## The 30-second version

- **Data:** the Kaggle Dog Breed Identification set, 10,222 labelled training images, 120 breeds.
- **Model:** MobileNetV2 (`mobilenet_v2_130_224`, TensorFlow Hub) as a frozen feature extractor, with one 120-way softmax layer on top.
- **Training:** Adam, categorical focal cross-entropy, 224x224 inputs, batch size 32, TensorBoard logging, early stopping on validation accuracy.
- **Best result:** 78.1% validation accuracy (run 6), after 19 epochs with early stopping.
- **Honest caveat:** 97.2% on training data against 78.1% on validation. It overfits.

## What the pipeline does

1. Reads `labels.csv` and the folder of JPEGs.
2. Splits 80/20 into training and validation sets.
3. Decodes each image, converts it to float32 and resizes it to 224x224.
4. Batches with `tf.data` (32 per batch, training set shuffled).
5. Trains the head on top of the frozen MobileNetV2.
6. Trains a final model on the full dataset, then predicts the Kaggle test set and a few dog photos of my own.

## Results: six runs

All runs use MobileNetV2 from TensorFlow Hub with a 120-way softmax output, Adam and categorical focal cross-entropy, 224x224 inputs, batch size 32.

| # | Data | Change | Train accuracy | Validation accuracy |
|---|---|---|---|---|
| 1 | 5,000-image subset | Baseline, 100 epochs | — | 77.1% (on a 1,000-image validation split) |
| 2 | 5,000-image subset | Augmentation: random rotation and brightness, 200 epochs | — | 75.2% |
| 3 | Full dataset | Pre-trained base set to trainable (fine-tuning) | — | 46.1% |
| 4 | Full dataset | Fine-tuning plus an extra ReLU dense layer | — | 45.9% |
| 5 | Full dataset | Removed the ReLU layer and augmentation, base still trainable | 80.4% | 44.4% |
| 6 | Full dataset (80/20 split) | **Base frozen** again | 97.2% | **78.1%** |

Runs 1 and 2 were measured on a smaller validation split than runs 3 to 6, so compare them with care. I did not record train accuracy for runs 1 to 4.

## Why fine-tuning hurt

- Making MobileNetV2 trainable dropped validation accuracy from about 77% to about 45%, and opened a big gap between training and validation accuracy. That is overfitting.
- With about 85 images per breed, there is not enough data to retrain the base without damaging what it had already learned.
- Freezing the base again (run 6) gave the best generalisation: 78.1% over 120 breeds.
- The remaining gap, 97% train against 78% validation, points to overfitting in the head and to too little data per breed.

## What I would do next

- Add augmentation to the frozen-base model. It only ran on the subset (run 2, where it did not help).
- Unfreeze only the top few layers with a low learning rate, after the head has converged.
- Try a larger backbone (EfficientNetV2 or ConvNeXt) and report top-5 accuracy.
- Keep one fixed validation split across all runs, and add a confusion matrix of the most-confused breeds.

None of this is done. Only top-1 accuracy has been measured.

## How I used AI (and what I did)

I can only vouch for what is written down, so this is the short, checkable version.

- **Mine:** the experiments and the comparison between them. Which runs to try, what changed in each, and the conclusion that the frozen base generalises best are recorded in my experiment log. The model and data are from the course and Kaggle, and the course gave me the starting point.
- **AI:** I used AI-assisted tooling later, to review the exported script. It surfaced the four small bugs listed below, and I had the fixes checked by a second reviewing pass. No training was re-run and no result changed.
- I have not written down any other AI contribution to this project, so I am not claiming any.

## Bugs I fixed later

Four small fixes to the exported script `src/dogscode.py`. No retraining, and the results above are unchanged.

1. **Typo:** `len(filnames)` became `len(filenames)`. It sat in the `else` branch of the file-count check, so the branch would have crashed with a `NameError` instead of printing.
2. **Wrong variable:** the `unbatch` helper looped over `val_data` instead of its own `data` argument. It now uses the argument.
3. **Plotting the wrong thing:** `plot_pred_conf` drew the bar chart from the label names instead of the prediction probabilities. It now plots the probabilities.
4. **Import shadowing:** `import matplotlib.image as image` was clobbered because `image` is reused as a variable all over the script. It is now `import matplotlib.image as mpimg`, and the one `imread` call uses it.

Still open, and I would rather say so: the script is a Colab export, so the `!pip install` line is a shell command and the file will not compile as plain Python. `predictions` is also only defined in the notebook flow, not in the script. I have not checked the notebook for the same four bugs, and I have not re-run the fixed script, because it needs TensorFlow and the dataset.

## Run it

1. Open `notebooks/DogsCode.ipynb` in Google Colab with a GPU runtime.
2. Download the [Kaggle data](https://www.kaggle.com/c/dog-breed-identification/data) and put it in your Google Drive under `Colab Notebooks/dog-breed-identification/` (`labels.csv`, `train/`, `test/`).
3. Run the cells in order. `requirements.txt` lists the packages (`tensorflow==2.15.0`, `tensorflow-hub`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`); the first cell installs TensorFlow 2.15.

The Kaggle dataset and the trained `.keras` models are **not** in this repo, so you have to download the data yourself. The script paths point at Google Drive locations from my Colab setup and need editing if yours differ.

## What is in the repo

| Path | What |
|---|---|
| `notebooks/DogsCode.ipynb` | The full Colab notebook with outputs |
| `src/dogscode.py` | The same code exported as a script (Colab cells, so it still has `!pip` and Drive paths) |
| `RESULTS.md` | The experiment log and conclusions |
| `requirements.txt` | Python packages |

## Stack

Python, TensorFlow 2.15, TensorFlow Hub, Keras, NumPy, pandas, scikit-learn, Matplotlib, Google Colab (GPU).
