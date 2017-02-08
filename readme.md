Using Deep Convolutional Neural Networks to Classify Music (TENSORFLOW) 

The objective here is to classify music samples into a specific genre using convolutional networks. There are three genres - Alternative, Rap/Hiphop, and Rock.

**Data set** 

* Music Audio Benchmark Data Set [Homburg et al. (2005) and Mierswa et al. (2005)]
* 294 song excerpts from each of 3 genres in mp3 format
* file frequency: 44.1 kHz
* file bitrate: 128 kb
* duration: 10 seconds each
* train : test data split:  80% : 20% (236 : 58)
* For each excerpts of the training data, we extracted 12 segments with length of 5 seconds to be a sample. In total 2832 training samples.
* For each 5 second signal, we extracted 22050 * 5 time points, with sample rate= 22050.  
* MFCC for each sample is of dimension 40 X 217

* Data has been processed beforehand, so the code here only load the data and apply the CNN for training and testing.

### CNN with tensorflow

3 convolutional layers and 2 fully connected layers, a normalization step before entering the fully connected layer. This implements with accuracy of 86%.

### Summary reported on tensorboard
* please change the log location ('/home/jidan/test/train') to your own. 