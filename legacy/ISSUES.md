# Issues found in the original project

The original project ([shsarv/Machine-Learning-Projects › Distracted Driver Detection](https://github.com/shsarv/Machine-Learning-Projects/tree/main/Distracted%20Driver%20Detection))
is preserved unmodified in [`original_notebook.ipynb`](original_notebook.ipynb). These are the problems found while rebuilding it, and how
v2 handles each one.

## Bugs that affect results

| # | Issue | Effect | Fix in v2 |
|---|-------|--------|-----------|
| 1 | **`K.set_learning_phase(1)`** is set globally at import time. | Every `evaluate()`/`predict()` call (dev scores, holdout predictions, the saved `e10.h5`) runs BatchNorm in *training* mode, using the statistics of each test batch instead of the learned running averages. Predictions depend on what else is in the batch, and the reported dev/holdout numbers are unreliable. | Standard `model.train()` / `model.eval()` switching. |
| 2 | **Double/inconsistent normalization.** `CreateImgArray` applies `imagenet_utils.preprocess_input` (Caffe mode: RGB→BGR + per-channel mean subtraction, values ≈ −124…151), and later the notebook divides by 255 again (`X_train = X/255`). | The inputs are BGR, roughly in [−0.5, 0.6], and not zero-centred per channel. The `(X == 0).sum()` "sanity check" is meaningless after mean subtraction, and so is the `Rescale()` helper used for display. | Images go to [0, 1] once, then are normalized with the backbone's own mean/std. |
| 3 | **The final model is trained and "validated" on the same data.** The holdout section fits on all 22,424 images with no validation set. Kaggle's test set is unlabeled, so the reported "holdout loss 2.64" cannot be reproduced. | There is no trustworthy estimate of generalization to new drivers. | A fixed, committed **driver-disjoint** split (18 train / 4 val / 4 test drivers). The model is selected on val and reported once on test. |
| 4 | **The submission file is invalid.** `np.savetxt("test_results.csv", probabilities)` writes no header and no image IDs. | It can't be submitted to Kaggle. | `python -m ddd.predict --submission ...` writes `img,c0..c9` with calibrated, clipped probabilities. |
| 5 | **The per-driver accuracy plot has mismatched labels.** `plt.bar(scores.index, scores['Test_acc'].sort_values())` sorts the values but not the index. | Bars are labeled with the wrong driver IDs, so the "p081 lowest / p002 highest" commentary isn't supported by the plot. | Per-driver accuracy is computed into a dict keyed by driver (`metrics_test.json`). |
| 6 | **Circular "sanity check" on predictions.** `PrintImage(X_holdout_scaled, 50000, Y=probabilities)` prints `y = argmax(probabilities)` and then `y_pred = argmax(probabilities)`. | The two always agree, so the check proves nothing. | Grad-CAM visualizations and a confusion matrix on labeled held-out drivers. |
| 7 | **Wrong arithmetic in the bias/variance discussion.** For example, "variance increased by 80%" compares relative percentages of an absolute gap, and the "Summary of Results" table mixes LOGO dev loss (3.79) with a full-data training loss (0.93). | The conclusions don't follow from the numbers. | Clean train/val/test curves with the same metrics on every split. |

## Reproducibility / engineering problems

* **Missing files.** `resnets_utils` is imported but not included. `driver_imgs_list/driver_imgs_list.csv` and `test_file_names.csv` are referenced but not part of the repo. The README lists `resnets_utils.py`, `requirements.txt` and `distracted_driver_detection.ipynb`, but none of these exist (the notebook is actually called `Distrated Driver detection.ipynb`).
* **Dead / removed APIs.** `keras.wrappers.scikit_learn`, `keras.utils.vis_utils`, `keras.utils.data_utils.get_file`, `scipy.misc` and `K.set_learning_phase` no longer exist in current Keras/TF/SciPy, so the notebook can't run on a modern stack. Many imports (`pydot`, `SVG`, `KerasClassifier`, `StratifiedKFold`, `cross_validate`, …) are unused.
* **Inefficient preprocessing.** It loads all images into a float64 array (22,424×64×64×3×8 B ≈ 2.2 GB, ≈ 7.8 GB for the 79,726 test images). Labels are fetched with an O(n²) `data.loc[data['img'] == current_img]` lookup per image.
* **Hard-coded fold count.** `cvscores = np.zeros((26, 4))` silently breaks if the driver count changes.
* **26-fold LOGO × 10 epochs** retrains a ResNet50 from scratch 26 times per configuration. The cost is too high to tune anything with, which the notebook itself identifies as the reason results are poor.
* **Truncated outputs.** The "10 epoch" training log shows 8 epochs.

## Modelling choices that limited accuracy

* **ResNet50 trained from scratch at 64×64 px** on images from only 26 people. Hands, phones and cups are a few pixels wide at that resolution, and a 23M-parameter network memorizes the drivers. The notebook's dev accuracy was 40.7%.
* **No augmentation, regularization or LR schedule.** When augmenting this dataset, **horizontal flips must be avoided**: they turn "texting – right" into something that looks like "texting – left" (c1↔c3, c2↔c4).
* **No calibration**, even though the competition metric is log loss.
