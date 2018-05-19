# Overwatch Vision Analytics

Jupyter notebooks with computer vision code that extract metadata out of Overwatch frames and videos.

## Strategies for improving Ult charge recognition 
* [DONE] gather more data, putting images in directories named after the ult charge
* train with more epochs
* augment the data by using translation since the bbox cropping may not always center the data
* vary the background so that it's not just one solid color, but also horizontal and vertical stripes
* account for the ult charge moving when the player jumps
  * to account for jumping, use a larger bbox to capture the ult charge, and remove the yellow ult meter with some image processing
