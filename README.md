# capstone

### Content
Data: In this folder, you can get the dataset required by the project, the data set includes news headlines, news body, pictures and news labels

Notebook: This folder contains the code of the project. All the data preprocessing in this project (including image enhancement, delete stop words, etc.) are included in the code file.
1. CNN_image_only_model.ipynb: Using CNN model to realize image-only classification
2. bert_text.ipynb: Using BERT model and main text feature to realize news classification
3. bert_title.ipynb: Using BERT model and title feature to realize news classification
4. Multimodal.ipynb: using multimodal neural networks to realize news classification

Image: This folder contains images from reports and blogs, including model structure and results

### Environment

It is recommended to reproduce the code on Google Colab, because the code may report errors locally due to version and hardware reasons

### Problem statement

With the development of the Internet, news content on the Internet has become more abundant. In order to improve the user experience, some major news portals will provide users with engaging content through recommendation systems. In this process, we need to use models to classify news accurately and in real-time. Nowadays, news classification mainly relies on models based on natural language processing, such as RNN, LSTM, and BERT. These conventional news classification models only consider textual information and ignore images, although images also contain essential news information. Therefore, we intend to use both image and text features to classify news through multimodal neural networks in this project. Besides, we compare the classification results of the multimodal neural network with the conventional models. The experiment result shows that multimodal methods can be a good trade-off between resource consumption and classification accuracy in the news classification task. The multimodal method can improve the model's classification accuracy from 0.97 to 0.98 with similar resource consumption.

### Dataset

In news classification, the BBC news dataset is commonly used.  Although it contains a limited amount of data, it provides labels for some subtasks such as classification and sentiment strength. 

In this project, we use this dataset. But on the original dataset, we only have the text and label information. To realize multimodal tasks, we need to crawl images that match the text on BBC’s official website. Next, we remove some news that does not have match images and news that cannot be found on the official website. Finally, our dataset contains 2026 pieces of data are divided into business, entertainment, politics, sport, and technology, and the number of each category is roughly balanced. Meanwhile, each piece of data contains a news headline, text body, and images.

### Methodology

We use the same data processing method as the image-only news classification task and the Text-only news classification task in the multimodal task.

Now the mainstream multimodal models are divided into dual-stream architectures and single-stream architectures. The main difference between them is that the single-stream architectures will directly connect the image and the text features at first and then train both features in a transformer model. In contrast, in the multi-stream architecture, we first process the text feature and image feature separately in different models, then concatenate the processed results of the text model and image model and use the concatenated features to get the final prediction result through the transformer.

<img width="473" alt="image" src="https://user-images.githubusercontent.com/69946337/166588638-a52b5403-fd37-4fde-baeb-9e5c2bf5a754.png">

In news classification, the information carried by text is usually more prosperous than that of images. If we use a single-stream model, the accuracy of picture information is relatively low, but when the weights of text information and image information are the same, the training effect of the single-stream model is poor because the image causes a high error. Therefore, we decide to use convolution neural networks to train images and use BERT to train text; then, we can collect a series of vectors representing image information and text information through the training of text-only and image-only models, as shown in Figure 1. Next, we put these vectors into the transformer model and get the final classification result through the transformer structure.!
In news classification, the information carried by text is usually more prosperous than that of images. If we use a single-stream model, the accuracy of picture information is relatively low, but when the weights of text information and image information are the same, the training effect of the single-stream model is poor because the image causes a high error. Therefore, we decide to use convolution neural networks to train images and use BERT to train text; then, we can collect a series of vectors representing image information and text information through the training of text-only and image-only models, as shown in Figure 1. Next, we put these vectors into the transformer model and get the final classification result through the transformer structure.

### Thanksgiving
[1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.

[2] ALBERT (from Google Research and the Toyota Technological Institute at Chicago) released with the paper ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.

[3] BERT (from Google) released with the paper BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

[4] VisualBERT (from UCLA NLP) released with the paper VisualBERT: A Simple and Performant Baseline for Vision and Language by Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang.

[5] Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. (Publisher link).

[6] Bradski, G. (2000). The OpenCV Library. Dr. Dobb&#x27;s Journal of Software Tools.
