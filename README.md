## Screen_Detector

# Project Structure : 

predict.py  -> Main file; python predict.py any_test_set_image.jpg. 

predictor.py  -> Loads MobileNetV2, extracts embeddings, applies trained classifier.
train.py  -> Extracts embeddings from dataset/, fits logistic regression, saves model_weights.json.
evaluate.py  -> Runs the trained model against dataset/ and prints confusion matrix + latency.

model_weights.json  —> Trained classifier weights.

# Approach :

1. Rather than hand-deriving features from first principles (moiré/FFT analysis, glare shape, texture uniformity, edge geometry), I extracted a 1280-dimensional embedding from **MobileNetV2**, pretrained on ImageNet with it's classification head removed. The CNN here is used purely as a fixed feature extractor not a Neural Network, identical in spirit to using a calculator someone else built.

2. A *logistic regression classifier* (scikit-learn, L2-regularized, C = 0.1) is then fit on top of these embeddings to separate real photos from screen recaptures. This two-stage design (frozen feature extractor + simple linear classifier) keeps the trainable part of the system tiny, fast, and easy to retrain on new data without touching the heavier CNN component.
   
4. I initially tried a *first principles approach* (FFT-based detection, glare, local texture uniformity, straight-edge detection) and tested individually, in pairs, and combined via stacking-style logistic regression. After two rounds of debugging (removing miscalibrated manual clipping thresholds, fixing feature scaling), this approach plateaued around 60-65% cross-validated accuracy even with different feature combinations or classifier choices(logistic regression, RandomForest, and SVM all converged to a similar accuracy value).
 
5. This indicated that handcrafted signals were too subtle or too dependent to generalize reliably from just 100 images. So, I changed the approach to pretrained embeddings.

# Results :

**Accuracy :**
*90.0% (+/- 6.3%) accuracy* under 5-fold stratified cross-validation on 100 self-collected images (50 real, 50 screen recaptures across phone, laptop, and TV displays, varied lighting, angle, and distance). This is the honest, held-out estimate; in-sample accuracy on the full training set is 100%, which I am explicitly not reporting as my accuracy, since that reflects memorization rather than generalization.
I expect some drop on SalesCode's held-out photos given differences in phone hardware, screen types, and lighting conditions not represented in my small dataset.

**Latency :**
*~215ms average, 383ms max per image* , measured end-to-end (Image load + CNN forward pass + Classification) on a laptop CPU (no GPU used).

**Cost per image :**
Cost on device is effectively 0 , the model runs locally on the user's phone with no server round-trip. MobileNetV2 is already designed for mobile deployment (~14MB), so this is the intended long-term path.

Cloud server : If we assume a single CPU core processing sequentially at ~215ms/image (~4.6 images/sec) on a low-cost instance (~$0.02/hour, e.g. AWS t3.small equivalent), cost works out to roughly **$0.0012 per 1,000 images**, or about **$1.20 per million images**. These are rough estimates assuming single-threaded CPU inference with no batching or GPU acceleration, both of which would lower cost further at scale.

# Improvements : 

1. More data; 100 images is very small for a held-out evaluation that needs to generalize across hardware I haven't seen, more images across more phone models and screen types would increase accuracy and probably get a better generalization.

2. I'd validate whether a smaller backbone (MobileNetV3-Small) maintains accuracy while cutting latency, since 215ms is acceptable but not "instant" for users on a phone.
 
**Adapting as cheaters adapt :**  Our embedding space captures broad structure rather than one narrow cue, hence it becomes harder to defeat with a single method.
I would definitely monitor Accuracy on newly collected fraud attempts over time and periodically *retrain the lightweight logistic regression layer* on fresh data without touching the CNN backbone.

**Phone-ready :**  MobileNetV2 is already suited well for on-device mobile inference. The natural path is converting it to Core ML for direct on-device execution, along with the logistic regression head, with no cloud dependency needed.

**Choosing the Cut-off:**  I think that *false negatives* (screen recapture thought as real) are more costly than *false positives*(flagging real photo for review), since false negatives lead to fraud/cheating. I'd choose a cut-off below 0.5 to reward recall on screen detection, then validate that choice against precision/recall tradeoffs on a larger held-out set rather than guessing. 
