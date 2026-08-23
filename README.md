\# Visual Quality Inspector



An AI system that looks at product surface images and tells you what's wrong with them — not just "defective" or "not defective," but the actual defect type, how confident it is, and (most importantly) \*where\* on the image it's looking to make that call.



Built this as a portfolio project to get hands-on with the full deep learning pipeline: dataset work, training a CNN from scratch, transfer learning, evaluation, and Grad-CAM explainability, then wrapping it all in a usable Streamlit app.



Worth saying upfront: this is a learning project, not a production inspection system. Real deployment would need way more data, more validation, and probably a lawyer.



\---



\## What it does



You upload an image of a steel surface, and the app:

1\. Runs it through a fine-tuned ResNet18

2\. Predicts which of 6 defect types it is

3\. Shows a confidence score (and flags it if confidence is low)

4\. Generates a Grad-CAM heatmap so you can actually see what part of the image the model based its decision on



That last part is the whole point. A model that just says "94% defective" isn't very useful in an inspection context — you want to know \*why\*.



\## Demo



\*(screenshots go here — add screenshots/home.png and screenshots/inspection\_result.png once you drop them in the screenshots folder)\*



\## Dataset



Used the \*\*NEU Surface Defect Database\*\* (NEU-DET) from Kaggle — 1,800 grayscale-ish steel surface images across 6 defect classes: crazing, inclusion, patches, pitted\_surface, rolled-in\_scale, and scratches. Perfectly balanced, 300 per class, all 200x200px, no corrupted files.



Split 70/15/15 (train/val/test) with a fixed seed so it's reproducible.



!\[Class Distribution](results/figures/class\_distribution.png)

!\[Sample Images](results/figures/sample\_images.png)



\## How it was built



\*\*Preprocessing\*\* — resized everything to 224x224 to match ResNet's expected input, normalized using ImageNet stats. Training images get light augmentation (flip, small rotation, brightness/contrast jitter) — nothing aggressive enough to make a defect look unrealistic. Validation and test images get no augmentation, since we want honest evaluation.



\*\*Baseline first\*\* — before jumping to transfer learning, I trained a simple 4-block CNN from scratch to have something to compare against. It did fine (96.67% test accuracy), but its main weak spot was confusing `pitted\_surface` with `inclusion`, which makes sense — both are speckled dark textures that look pretty similar at a glance.



\*\*Then transfer learning\*\* — swapped in a pretrained ResNet18 and fine-tuned the whole thing (not just the last layer) since these industrial images are pretty different from ImageNet's natural photos. Went with ResNet18 over something bigger like ResNet50 mainly because the dataset is small — a heavier model felt like it'd just overfit faster without much upside.



The difference was pretty noticeable — 99.63% test accuracy, and it got there in fewer epochs than the baseline did.



\## Results



| Model | Test Accuracy | Precision | Recall | F1 |

|---|---|---|---|---|

| Baseline CNN | 96.67% | 0.967 | 0.967 | 0.967 |

| ResNet18 (transfer learning) | \*\*99.63%\*\* | \*\*0.996\*\* | \*\*0.996\*\* | \*\*0.996\*\* |



\*\*Training curves:\*\*



Baseline CNN:

!\[Baseline Training Curves](results/figures/baseline\_cnn\_training\_curves.png)



ResNet18:

!\[ResNet18 Training Curves](results/figures/resnet18\_training\_curves.png)



\*\*Confusion matrices\*\* — you can see the baseline's confusion between pitted\_surface/inclusion pretty clearly, and how much cleaner ResNet18's predictions are (only 1 mistake out of 270 test images):



Baseline:

!\[Baseline Confusion Matrix](results/figures/baseline\_cnn\_confusion\_matrix.png)



ResNet18:

!\[ResNet18 Confusion Matrix](results/figures/resnet18\_confusion\_matrix.png)



\## Grad-CAM



This is the part I found most interesting to actually look at. For every class, the heatmap lines up with where the actual defect is in the image — not some random background region, which is what you'd worry about if the model was cheating on some unrelated correlation.



!\[Grad-CAM Examples](results/figures/gradcam\_all\_classes.png)



One caveat worth being upfront about: Grad-CAM gives you a rough "the model was looking around here" region, not a pixel-precise outline of the defect. Treat it as a general area of interest, not a segmentation mask.



\## Running it yourself



```bash

git clone https://github.com/HussnainAnjum28/visual-quality-inspector.git

cd visual-quality-inspector



python -m venv venv

venv\\Scripts\\activate        # Windows

\# source venv/bin/activate   # Mac/Linux



pip install -r requirements.txt

```



Then to launch the app:



```bash

streamlit run app/streamlit\_app.py

```



Opens at `http://localhost:8501`. Upload an image and it'll do the rest.



To run the test suite:



```bash

pytest tests/test\_app.py -v

```



If you want to retrain from scratch, the full pipeline (EDA → baseline → transfer learning) was done in Google Colab using a free T4 GPU — notebooks are in `notebooks/`. Grab the NEU dataset from Kaggle first.



\## Where this falls short



Being honest about the limitations here:



\- 1,800 images is a small dataset. It works great on this specific data, but I have no idea how it'd hold up on real factory images with different lighting, camera angles, or steel surfaces it hasn't seen.

\- It's trained only on steel surface defects — won't generalize to fabric, PCBs, or anything else without retraining on that kind of data.

\- There's no "normal / no defect" class right now — it always picks one of the 6 defect types, even on a clean image. That's a gap I'd want to fix before calling this anything close to production-ready.

\- Grad-CAM's localization is coarse, as mentioned above.

\- This has not been validated for anything resembling real industrial deployment — it's a demonstration of the ML pipeline, not a certified inspection tool.



\## What I'd add next



\- A "no defect / normal" class for actual pass/fail screening

\- Something like segmentation for tighter defect boundaries instead of a heatmap

\- Real-time inference from a live camera feed

\- Model quantization to make it lighter for edge devices



\## Stack



Python, PyTorch, torchvision, Streamlit, scikit-learn, pytorch-grad-cam, Matplotlib, Pandas. Trained on Google Colab (T4 GPU).



\## License / dataset credit



Built for educational/portfolio purposes. Dataset is the \[NEU Surface Defect Database on Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) — check that page for its license terms.



