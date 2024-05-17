No-frills implementation of a NN to classify MNist.

My model architecture (customizable)  
  Layers: two dense layers, of size 64 and 10 (with normalization)  
  Activation functions: leaky relu & softmax (for final layer)  
  Loss: cross-entropy (also have RMS and MSE)  

Trained for 1 epoch over 30 batches (2000 images each)

Training accuracy (max): ~90%  
Testing accuracy: ~85%

I could have definitely got the accuracy up with some more training, but it was taking far too long. My backprop implementation didn't seem super efficient, but it was really good to implement the entire thing from scratch.  
The model wasn't working at all to start with, so it was great to get under-the-hood & diagnose what was going wrong. That, and implementing backprop were the most valuable parts of this project. Glad I could get this done & excited for what comes next :)
