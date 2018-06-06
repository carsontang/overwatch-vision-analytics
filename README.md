# Overwatch Vision Analytics

Jupyter notebooks with computer vision code that extract metadata out of Overwatch frames and videos.

## Installation
`conda env create -f environment.yml`

## Strategies for improving Ult charge recognition 
* [DONE] gather more data, putting images in directories named after the ult charge
* train with more epochs
* augment the data by using translation since the bbox cropping may not always center the data
* vary the background so that it's not just one solid color, but also horizontal and vertical stripes
* account for the ult charge moving when the player jumps
  * to account for jumping, use a larger bbox to capture the ult charge, and remove the yellow ult meter with some image processing
* go through this checklist: https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

## Flaws
* validation data isn't uniform. Not all numbers are completely within the 28x28 box, nor are they centered. Jumping hasn't been factored in.
* train and validation data might not be prepared the same way. They need to be preprocessed consistently.
* only training with 8 epochs, try more to prove that the network is overfitting.
* RGB train's canvas needs to be different from the text
* [DONE] RGB train data isn't in the range 0-1
* use Gaussian blur to make data more realistic
* look at images that are being predicted correctly
* shuffle the data before training
