# House-GANs-Reproduction
Reproduction Code of [House-GAN: Relational Generative Adversarial Networks for Graph-constrained House Layout Generation](https://arxiv.org/abs/2003.06988)
Original code can be found in the Nelson Nauata's (first author) [github page](https://github.com/ennauata/housegan).

### What parts are reproduced?
Specific goals regarding the reproduced parts will be added soon.

### Why did we do this?
Well, first of all it's a great paper and an interesting read.
Secondly, it is a project for the course [CS4240 Deep Learning](https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=57391&_NotifyTextSearch_) in TU Delft.

### How to train
1. Get the original data from [here](https://www.dropbox.com/sh/p707nojabzf0nhi/AAB4UPwW0EgHhbQuHyq60tCKa?dl=0)
2. Make sure the dataset is in the correct path.
3. The entry point is `train.py`. Check that the arguments are correct. 
4. Enjoy your 24h waiting time.

### Epochs/Iterations
The original author defines one epoch as one pass on the whole dataset and as iteration the pass on a single batch (whose size is adjustable). According to the paper the training was done on 50-70 epochs or 200k iterations.

### Running tests
Run `python -m unittest discover test`
