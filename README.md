# bytes_cv
## Cool project that lets people take pictures of food ingredients to brainstorm recipes that they can make.

### What I did:
* leveraged PyTorch's FasterRCNN w/ ResNet50 backbone as a pretrained model, tweaked feature classification layer for custom object detection.
* Gathered 40k images from Google's Open Images Dataset of food related items, trained model on these items (90 classes)
* Interface to interact with application can be found [here](https://bytes.ndlug.org). Here is the [source code](https://github.com/kylephan5/bytes) for that website.
* Also can use this repository to interact w/ EdamamAPI (might be better recipe selection here!)

### Future work:
* I want to possibly detect custom classes outside of Open Images Dataset, and perhaps increase the precision of the current model. I want to also re-tweak what classes I am using from (COMING SOON!)
