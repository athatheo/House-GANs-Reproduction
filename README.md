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


Evaluation
------
In order to evaluate diversity, the FID score from https://github.com/mseitzer/pytorch-fid is used. One way to calculate the FID score is to generate on esample for each graph (5k fake) and compare it with the corrresponding GT (5k real).
So in order to evaluate all 5 metrics provided, to evaluate fro each group the model should be trained with the data of the other groups. Ex. to evaluate 1-3, we need to train a model on 4-6, 7-9, 10-12, 13+.

For compatibility evaluation run python compatibility_figure.py
