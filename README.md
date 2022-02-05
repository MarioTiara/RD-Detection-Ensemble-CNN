# Retinopathy Diabetic (RD) Classifier - Ensemble CNN
<p align="justify">
Detect the stage of diabetic retinopathy in human retina through fundus images. The dataset that used is from 4th Asia Pacific Tele-Ophthalmology Society (APTOS) in <a href="https://www.kaggle.com/c/aptos2019-blindness-detection">Kaggle </a>. The total of images we used are 8000 images, which is 6000 for training, 1500 as data validation and 500 for testing. We used several proprocessing method for getting hight performance of our model. Our final model is an ensemble of 3 convolutional neural network model with different convolutional network backbone (DensNet201, InceptionV3 and MobileNetV2). Complete project with a simple GUI made with python and PyQt5. Image below show our project framework looks like.
  <p align="center">
    <img src="https://user-images.githubusercontent.com/43440326/152648046-255287bb-fda8-406e-b40a-35d581454569.png" width="767" height="430"/>
  </p>
 </p>
<h2 style="font-size:10vw">Experimental Results</h2>
<p>
  <h3 style="font-size:10vw">Backbone performance in SGD and Adam Optimizer with learning rate variance</h3>
  <img src="https://user-images.githubusercontent.com/43440326/152647307-a157f81b-4714-4e39-a8ea-c076da35b1ee.png" alt="Sublime's custom image"/>
  Best accuracy of DensNet201 is 0.94 using Adam optimizer with 1.125e-5 lr, InceptionV3 is 0.93 using Adam with 1.25e-5 lr and MobileNet is 0.9 using Adam and 2.5e-5 lr.
  <h3 style="font-size:10vw">Comparing Single model with Ensemble model in different input</h1>
  We evaluate our model with 5 input variances, original and 4 variances of K of K-meanse clustering. 
  <img src="https://user-images.githubusercontent.com/43440326/152647701-89f01066-d8a8-465b-af10-31c4aeab0ce6.png" alt="Sublime's custom image"/>
  <h3 style="font-size:10vw">Detail Performance of Ensemble Model Using Confusion Matrix</h3>
  <img src="https://user-images.githubusercontent.com/43440326/152647787-d17d3193-db04-4d86-ba56-5d8cc0c6ddee.png" alt="Sublime's custom image"/>
  <img src="https://user-images.githubusercontent.com/43440326/152647844-bc83447a-def2-4fcb-8680-ac69ddd08df0.png" alt="Sublime's custom image"/>
</p>

### Demo
<img src="https://user-images.githubusercontent.com/43440326/152631586-f3b13b5b-9ae0-4fc5-b201-b4b94dbffa8e.gif" alt="Sublime's custom image"/>

<h3 align="justify">
The Research paper "NON-PROLIFERATIF RETINOPATHY CLASSIFICATION USING ENSEMBLE CONVOLUTIONAL NEURAL NETWORK" is publised in Pseudoce jurnal Univerity of Bengkulu <a href="https://ejournal.unib.ac.id/index.php/pseudocode/article/download/13619/7272">paper link</a>
</h3>



