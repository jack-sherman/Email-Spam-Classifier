# Email Spam Classifier
In this project I use Gaussian Naive Bayes to calssify spambase data from the UCI machine learning resository. This project was completed while I was enrolled in a machine learning course at my university.

# Method
The first thing that I did with the dataset was split it into 2 parts corresponding to 'spam' and 'not spam'. I took these 2 parts and combined half of the data from each to create 2 equally sized data sets with about 906 spam and about 1394 not spam data points.
![image](https://user-images.githubusercontent.com/47011094/156500258-25e50065-b418-42f2-a2fc-92fdb6cf51d5.png)
With the dataset split into a training and test set, I create a proabilistic model using the training data. I first calculate the mean of each attribute in the training set, and then calculate the standard deviation of each attribute. I then replace each of the standard deviations equal to 0 with a min value in order to avoid dividing by zero.
 Once I find the std deviation and mean, I use the equations below in order to classify each of the test points.
 ![image](https://user-images.githubusercontent.com/47011094/156500443-929306d0-4b2a-462f-a804-c1bed87b3516.png)
In this specific implementation of GNB, I'm using logarithms to avoid underflow.
# Results:
![image](https://user-images.githubusercontent.com/47011094/156500532-4af9ed20-26c0-4a7c-b7fa-024fd605d9cd.png)

  Accuracy = 1771/2302 = 0.769
  
  Precision = 834/(834+455) = 0.647
  
  Recall = 834/(834+72) = 0.921

Confustion matrix:

![image](https://user-images.githubusercontent.com/47011094/156500657-33980091-8388-40a7-a5d1-3e9c3a35d904.png)

I do think that all of these attrivbutes are independent because the algorithm did a good job at detecting actual spam. Despite the attributes being independent, Naive Bayes did poorly in terms of overall precision and accuracy. The algorithm mainly did badly when classifying 'not spam' data. False positive in this case are extremely bad. A user wouldn't want important emails to get marked as spam by their spam filter. Onbe reason that Naive Bayes might be performing poorly is because there could be attrivutes that are common in non-spam emails that are also consistently present in spam emails. Certain attributes like long uninterrupted strings of capital letters make iut fairly easy to determine if an email is spam. The dataset provider also says that there exists some misclassification error in the dataset itself, but not in a large enough amount where the accuracy should be this low. Perhaps with more data these tests could have gone better.
