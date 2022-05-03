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

It is recommended to accomplish the code on google colab, because the code may report errors locally due to version and hardware reasons

### Problem statement

With the development of the Internet, news content on the Internet has become more abundant. In order to improve the user experience, some major news portals will provide users with engaging content through recommendation systems. In this process, we need to use models to classify news accurately and in real-time. If you were a data scientist for a news portal, what method would you use to categorize news? Usually, we will use some conventional natural language processing models to deal with the problem of news classification, such as CNN, LSTM, and BERT. However, these traditional news classification models only consider textual information and ignore images, although images also contain essential news information. Will this achieve a better result if we use a multimodal neural network to combine image and text features? In this blog, we will discuss the performance of multimodal neural networks on news classification and compare it with the conventional models.

### Method


