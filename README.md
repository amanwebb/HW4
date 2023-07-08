# Homework Summary

For the given homework prompt, the goal was to train a model on the full MNIST dataset (digits 0-9) and create a handwritten dataset consisting of five characters: A, B, C, D, and E. The trained model should then be transferred to classify the images in the created dataset.



## Image Preprocessing and Data Generation

The image preprocessing and data generation process for this homework consisted of the following:

1. **Loading the MNIST dataset:**
   - The MNIST dataset is loaded using the `mnist.load_data()` function from Keras. It provides a training set `(x_train, y_train)` and a test set `(x_test, y_test)`.
   - This was taken from the code provided in the assignemnt. I edited it to include all numbers (0-9) rather than it split into two groups liek the example had it.

2. **Loading letter images:**
   - I stored all of my images in a folder and created a file path to that folder.
   - A loop was used to iterate over each letter ('A' to 'E') and each variation (1 to 10) of the letter. Meaning that I had 50 total images. 
   - Inside the loop, the image path for each letter and variation is constructed using `os.path.join()` with the folder path and the current letter and variation number.
   - The image is opened using `Image.open()` from the PIL library and converted to grayscale (`'L'` mode) since the MNIST dataset consists of grayscale images. I learned that the hard way... 
   - The image is then resized to the desired dimensions (`img_rows` and `img_cols`) using the `resize()` method of the PIL `Image` object.
   - The resized image is converted to a NumPy array using `np.array()` to obtain a 2D array representing the pixels.
   - To match the input shape required by the model, an additional dimension (corresponding to the channels) is added to the array using `np.expand_dims()` with `axis = -1`. This was probably the worst bug I experienced. I repeatedly received errors similar to `Shapes do not match`
   - The processed image array and its corresponding label (index of the letter in the `letters` list) are appended to separate arrays (`x_train_letters` and `y_train_letters`, respectively).

3. **Preprocessing character datasets:**
   - The character dataset images (`x_train_letters`) are normalized by dividing by 255.0 to bring the pixel values in the same range as the MNIST images. Somewhat copied the code provided in the assignment.

4. **Combining datasets:**
   - The MNIST dataset images (`x_train`) and letter dataset images (`x_train_letters`) are concatenated along the first axis (samples) using `np.concatenate()`. This combines both sets of images into a single array (`x_train_combined`).
   - The same operation is performed for the labels (`y_train` and `y_train_letters`) using `np.concatenate()`. This creates a combined set of labels corresponding to the combined images (`y_train_combined`).

The data generation process results in combined training data (`x_train_combined` and `y_train_combined`) containing both the original MNIST dataset images and the additional letter images I genreated through actual handwriting. This combined dataset allows the model to learn from both types of images during training and enables the classification of both digits and letters (A-E).   

## Bugs

1. **Issues and bugs faced along the way**
   - I was presented with many obstacles along the way:
        - imprting the handwritten dataset
            - This was just a file path issue that was actually easy to solve after taking a short break
        - Number of classes were not equal
            - I believe this error happened due to having 10 digits and only 5 characters
        - tensorflow detecting repeated tracing of a function
            - This one was confusing, but somehow (with the help of chatgpt) it was resolved
        - I experienced so many error warnings with shapes not being compatible    
            - This was mentioned earlier with the `shapes do not match`
            - eventually learned to add another dimension


## Sumamry adn Conclusion
Completing this homework assignment was a challenging yet rewarding experience. Despite encountering various difficulties, including errors, bugs, and the need to understand and modify existing code (the one given in the actual assignment), I persevered and successfully trained a model that could classify both digits and characters (A-E). The resulting classification model came out with approximately a 40-45% accuracy score. more improvements can be made as I look into this a little deeper. The process involved importing the MNIST dataset, creating a handwritten character dataset, preprocessing and combining the datasets, and training the model.

The journey was not without its obstacles, such as file path errors, class imbalance, shape mismatches, and TensorFlow warnings. However, through diligent debugging and troubleshooting (not to mention hair pulling and crying), I overcame these challenges (with the help of stackoverflow and chatgpt) and gained valuable insights into problem-solving and code adaptation.

The sense of accomplishment was immense as I witnessed the epochs running, saw the accuracy improving over time, and realized that my efforts were paying off. It was a testament to the power of determination and perseverance in the face of complex assignments.

Despite the lack of sleep endured during this process and the time constraints involved, the completion of this assignment provided invaluable learning experiences. It deepened my understanding of machine learning, reinforced my problem-solving skills, and fostered a sense of achievement that comes with overcoming obstacles.
 






