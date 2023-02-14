# Baxtableep

### What is it?
Baxtableep is a profanity detection module for python orignally built for <a href='https://github.com/rabbitminers'>Baxter</a> to help quickly and effectively manage frequent user messages without using a set wordlist allowing for far more consistant detections.

In place of an explicit blacklist like that used in most profanity detection library Baxtableep has been trained on a large dataset to identify similarities between profane strings. If you wish you can utilise your own dataset and train your own model with the included training scripts

### Usage
Note 'Offensive Text' is a substition for text containing profanity

```python
from baxtableep import Baxtableep

bleep = Baxtableep()

bleep.predict(['Offensive Text', 'Normal Text'])
# [1, 0]

bleep.predict_probs(['Offensive Text', 'Other Text', 'Other Offensive Text'])
# [~0.9, ~0.1, ~0.8]
```
