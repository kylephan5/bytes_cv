# bytes_cv
## Cool project that lets people take pictures of food ingredients to brainstorm recipes that they can make.

### What I did:
* leveraged PyTorch's FasterRCNN w/ ResNet50 backbone as a pretrained model, tweaked feature classification layer for custom object detection.
* Gathered 40k images from Google's Open Images Dataset of food related items, trained model on these items (90 classes), leveraged [Fifty-One](https://docs.voxel51.com/user_guide/using_datasets.html) for datasets
* Interface to interact with application can be found [here](https://bytes.ndlug.org). Here is the [source code](https://github.com/kylephan5/bytes) for that website.
* Also can use this repository to interact w/ EdamamAPI (might be better recipe selection here!)

### Future work:
* I want to possibly detect custom classes outside of Open Images Dataset, and perhaps increase the precision of the current model. I want to also re-tweak what classes I am using from (COMING SOON!)

### Thanks to:
* Professor Flynn for helping me out with some of the kinks that I had re: Fifty-One and my project. Also thanks for letting me use your GPUs to train my models :D
