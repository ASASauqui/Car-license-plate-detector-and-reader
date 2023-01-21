<h1><b>Car license plate detector and reader</b></h1>

Vehicle license plate detector and reader that only uses Machine Learning techniques and image processing.<br>

<div align="center">
    <img width="50%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/methodology/methodology_8.png?raw=true" />
    <p>Plate prediction.</p>
</div><br>

<h2><b>Introduction</b></h2>

The use of support vector machines, hierarchical clustering, contour detection using external recovery and recovery trees, and the threshold adaptive mean algorithm for the detection and reading of license plates will be presented. Our method combines various image processing techniques for robust plaque detection and makes use of two support vector machine models for letter and number prediction.<br>
The problem to solve is about the detection of car license plates and their subsequent reading and interpretation using only Machine Learning (ML) techniques and image processing, something complicated due to the fallibility of some Machine Learning techniques compared to other more complex branches. , as are Deep Learning and Artificial Intelligence. Various proposals have appeared making use of these last two branches to solve these types of problems, but it was decided to make a methodology based on pure ML, which obtained more than acceptable results in terms of prediction of car license plate characters, being able to detect and read plates with little contrast or that are somewhat saturated with visual information.<br>
We proposed the use of two support vector machine models for the prediction of letters and numbers based on probability; image segmentation using image retrieval and processing trees; plaque detection using recovery structures and adaptive threshold mean; search for rectangles from contours; and the use of hierarchical grouping for the detection of similar contours through points of interest extracted from the rectangles that enclose said contours.<br><br>

<h2><b>Used technologies</b></h2>

It is a Machine Learning project made with:

<table align="center">
    <tr>
        <th align="center" style="text-align: center">
            Programming Language
        </th>
        <th align="center" style="text-align: center">
            Libraries
        </th>
    </tr>
    <tr>
        <td align="center">
            <img src="https://img.shields.io/badge/python-blue.svg?style=for-the-badge&logo=python&logoColor=white">
        </td>
        <td align="center">
            <img src="https://img.shields.io/badge/sklearn-c1ae32.svg?style=for-the-badge">
            <img src="https://img.shields.io/badge/scipy-white.svg?style=for-the-badge&logo=scipy&logoColor=071a68">
            <img src="https://img.shields.io/badge/cv2-ab33c6.svg?style=for-the-badge">
        </td>
    </tr>
</table>
<br><br>

<!-- <h2><b>Demo (not available yet)</b></h2>

Currently there is no live visualization, since it is planned to migrate the database to MongoDB and make a modification to the Front-End to improve the view, which is currently very basic. As well as switching from JavaScript to TypeScript.
<br><br> -->

<h2><b>Installation and running (local)</b></h2>

### 1. Clone the repository
```
git clone <url>
```

### 2. Running and installations
Using Jupyter Notebook you will be able to open and experiment with any of the 5 code with .ipynb extension, and you will be able to use the code with .py extension using normal Python.

Make sure you have Scipy, Sklearn, CV2, etc. installed in order to use the codes.
<br><br>

<h2><b>Codes division</b></h2>

You will find a total of 6 codes made in Python, of which 5 have the .ipynb extension, and only 1 with the .py extension. Each one is independent and has different functionality.

<ol>
    <li>
        <b>01 - Datasets Creation.ipynb:</b>
        In this code, the creation of the datasets that will be used to train the final models was made, here all the necessary image processing was done so that the model is very clear on how to classify values.<br><br>
        <div align="center">
            <img width="70%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/codes_images/1.png?raw=true" />
            <p>Dataset information in .csv.</p>
        </div><br>
    </li>
    <li>
        <b>02 - Models Creation.ipynb</b> In this code, the creation of the support vector machine models was carried out.<br><br>
        <div align="center">
            <img width="50%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/codes_images/2.png?raw=true" />
            <p>Model files.</p>
        </div><br>
    </li>
    <li>
        <b>03 - Methodology - Experiments (Only Car Plate).ipynb</b> In this code, the methodology for the detection and reading of automobile license plates was applied when only an image of the license plate is given, without the car.<br><br>
        <div align="center">
            <img width="40%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/codes_images/3.png?raw=true" />
            <p>Detection and reading of important components.</p>
        </div><br>
    </li>
    <li>
        <b>04 - Methodology - Experiments (Car with Plate).ipynb</b> In this code, the methodology for the detection and reading of car license plates was applied when an image of a car with its license plate is delivered.<br><br>
        <div align="center">
            <img width="40%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/codes_images/4.png?raw=true" />
            <p>Detection and reading of important components in a car plate with car.</p>
        </div><br>
    </li>
    <li>
        <b>05 - Methodology - Plate detection on video.py</b> In this code, the methodology for the detection and reading of car license plates was applied, but in real time, through a video that was provided. It usually goes slow.
    </li>
    <li>
        <b>Random experiments.ipynb</b> This code was the mother code, from here the others arose, since many of the experiments necessary for the methodology to work were carried out in it. It doesn't have any logic, they are just tests and tests, but we decided to leave it here because it was the beginning of everything.
    </li>
</ol>
<br><br>

<h2><b>Folders division</b></h2>

The 5 folders that you will be able to see in the project store: training and experimentation images, explanatory documents in a professional manner, and Support Vector Classifier (SVC) models created for the identification of letters and numbers.

<ol>
    <li>
        <b>Experiments datasets:</b> Inside this folder are images that were experimented with once the project was finished. They are images of car license plates, either in a car or individually, they were used in order to test the efficiency of the proposed methodology and verify that it works optimally.
    </li>
    <li>
        <b>Explanation document:</b> Inside this folder is an IEEE manuscript, which explains in a concrete way how this methodology for plate detection was created. If you want to know more about the Machine Learning techniques used, the creation of models, results, etc., read this manuscript.
    </li>
    <li>
        <b>Image datasets:</b> Inside this folder you will see that there are 4 folders that contain thousands of images of letters and numbers, they are all the images used for the creation of the final datasets for the identification of the letters and numbers of the car plates. Each of these images underwent image processing to improve the identification of values in the model.
    </li>
    <li>
        <b>Letters and numbers datasets:</b> Inside this folder there are several .csv files, which have the information of all the images of numbers and letters that went through an improvement process for their greater identification in a model. These are the datasets that the letter and number models were fed with.
    </li>
    <li>
        <b>Models:</b> Here you simply find the finished templates ready to use for number and letter identification.
    </li>
</ol>
<br><br>

<h2><b>Methodology</b></h2>

Below is a full explanation of the methodology used.

<h3><b>1. Preparation of datasets and models for the prediction of letters and numbers</b></h3>
<ul>
    <li>
        <b>A. Datasets</b><br>
        First, two datasets were obtained: one of 26,400 images of letters of the English alphabet, and another of 10,100 images of numbers. But, for the application of our proposal, it was decided to apply an image processing to these datasets, since each image contained too much excess space that would affect the quality of the models' predictions. Image processing for the datasets consisted of: converting the image to grayscale; then apply an Otsu threshold to be able to binarize the image and distinguish the objects in a better way; search for contours within the image using external retrieval for the detection of the largest block of pixels (which in this case is the specific character in the image); search for the largest rectangle in the image (in case there is some kind of unwanted segmentation); Up to this point, all this was done to only identify the character that matters to us within the image and crop it to eliminate the excess space from it. Once the new cropped image is obtained, it must be resized to 28x28 pixels; apply grayscale again (since the cropped image was obtained directly from the image without processing); apply Gaussian blur to correct blemishes within the image; and, finally, reapplying an Otsu thresholding to obtain only black or white values in the image pixel matrix (0 or 255).<br><br>
        <div align="center">
            <img width="60%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/methodology/methodology_1.png?raw=true" />
            <p>Image of the letter 'A' after going through image processing.</p>
        </div><br>
        Once the appropriate image processing has been applied, the image was converted to a single dimension, these values will be our 'x' in the training of our model, and the 'y' will be the ASCII code of said character. Therefore, this ASCII code must be added to the end of the new one-dimensional vector created earlier. Each image is added to a list, whether it is a letter or a number, here there is a division, because there will be a model exclusively for letters and another for numbers. Each of these two lists were randomized in position to obtain scrambled samples; each list was converted into a Dataframe and exported as a “.csv” file.<br>
        In this way, two datasets have been created, one exclusively for letters and the other for numbers.<br>
        These datasets will be useful for training our models for character prediction.
    </li>
    <li>
        <b>B. Support Vector Machine Models</b><br>
        It was intended to use multinomial logistic regression models or support vector machine models for character prediction, since these are dedicated to classifying elements of different classes in a vector space, but in the end the use of multinomial models was chosen. support vector machines for obtaining better accuracy and R2 results. In the case of the model for the letters, an accuracy of 0.908 and an R2 of 0.844 were obtained; and in the case of the number model, an accuracy of 0.949 and an R2 of 0.872, while the results of the logistic regression models maintained lower values than those previously presented. Support vector machine models, based on these statistical metrics, revealed that their predictions are good and maintain a “actual values-predictions” relationship close to the form of the function y=x.<br>
        As already mentioned in the previous section, two models were created, one exclusively for letters and the other for numbers, this to obtain better discernments between characters of the same species (letters or numbers), and to be able to make comparisons between similar characters and reduce the margin of error of prediction between them. For the creation of each model, the information of said datasets was segmented into 'x' and 'y', where 'x' are the binarized values of each image, and 'y' is the ASCII value that corresponds to said image. These values were entered into the support vector machine model to train it and later the model was exported to use it in the methodology for reading car license plates.
    </li>
</ul>

<h3><b>2. Plate detection</b></h3>
<ul>
    <li>
        This is the first part of the methodology for capturing and reading plaques, plaque detection; For this, the following methodology was carried out.<br>
        You need a frame or an image where a car appears that contains some type of visible license plate. To extract the plate from it, various image processes will be applied to the image to obtain important information that helps its identification.<br>
        First, the image is converted to grayscale; followed by the application of a bilateral filter, for the elimination of image noise and its smoothing; after this, a Gaussian blur was added to correct imperfections within the image; and once the pertinent corrections were applied, the threshold adaptive average algorithm was included, which detects important information of the image in the form of edges.<br>
        Once the edges were located, a contour search was applied using the list recovery technique (it shows all possible contours); These contours are ordered from largest to smallest according to their area and only the 25 largest contours are chosen to eliminate unnecessary ones.<br>
        For each contour, its area is drawn and it is enclosed in a rectangle for its detection. The perimeter of the original contour is calculated to approximate the shape of its figure, here the objective is to find figures that have 4 vertices (whether rectangles or squares), therefore, if the approximate figure contains only 4 vertices, and if the area of said figure is less than or equal to 50% of the area of the original figure, and if its aspect ratio is greater than or equal to 1.7 and less than or equal to 5 (aspect ratio that license plates usually have), then it will be considered that said rectangle contains a license plate, and the sector of said rectangle will be extracted from the original image to obtain the license plate and frame it.<br><br>
        <div align="center">
            <img width="60%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/methodology/methodology_2.png?raw=true" />
            <p>Plate detection algorithm framing the car plate in red.</p>
        </div><br>
    </li>
</ul>

<h3><b>3. Plate reading</b></h3>

Once the license plate has been identified and its image has been obtained, it can be read to find out what its serial code is.

<ul>
    <li>
        <b>A. Models importation</b><br>
        For plate reading it is important to import the models previously created for the prediction of letters and numbers.
    </li>
    <li>
        <b>B. Image processing</b><br>
        Again, the essential step in any image information reading process is image processing. The plate image will be converted to grayscale, a Gaussian blur will be applied to remove imperfections from said image and finally an Otsu thresholding to be able to binarize the image and distinguish the objects within it in a better way.<br><br>
        <div align="center">
            <img width="60%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/methodology/methodology_3.png?raw=true" />
            <p>Car license plate after image processing.</p>
        </div><br>
    </li>
    <li>
        <b>C. Obtaining important components</b><br>
        Having the image processed in an ideal way, we will proceed to search for contours using the recovery tree technique to obtain all the contours of the image; and a rectangle will be added to these contours that covers them in the best possible way.<br>
        The problem now is that there are too many rectangles for having obtained all the rectangles in the image, so the next task is to reduce the number of rectangles considerably, eliminating those that are not greater than or equal in width to 2% of the width of the image. original image and not less than or equal to 30% width of the original image; and at the same time, they must comply with being greater than or equal to 30% of the height of the original image and being less than or equal to 80% of the height of the original image. In this way, up to 95% of the rectangles initially obtained are usually eliminated (since there are usually dozens of rectangles the size of a single pixel) and only those that have a thin shape in width and have an elongated height remain, basically , the figure that letters and numbers usually have on a car plate.<br>
        Those rectangles that managed to meet the desired specifications, the image contained within them is extracted from the original image and resized to 28x28 pixels (the size of the images that our models accept). These images and rectangles are saved in variables for later use in some remaining procedures.<br><br>
        <div align="center">
            <img width="50%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/methodology/methodology_4.png?raw=true" />
            <p>Rectangles that met the specifications.</p>
        </div><br>
    </li>
    <li>
        <b>D. Obtaining points of interest from rectangles</b><br>
        This step is extremely important for the identification of the correct rectangles (by correct we refer to those rectangles that contain a letter or number and that belong to the serial code of the plate).<br>
        It was decided that each rectangle would have 3 points that are of interest to us for the identification of those that are suitable, which are: the 'y' coordinate of the point of origin of the rectangle (the vertex of the upper left side); y-coordinate of the vertex below the point of origin (the vertex of the lower left side of the rectangle); and finally the height of the rectangle. These points are of great interest, since normally the numbers and letters that are part of the plate serial code tend to have the same height and are usually aligned horizontally in a straight line, so those that meet these similarities have high probabilities of being part of the characters that make up the serial code.<br>
        The points of interest of each rectangle were saved in a list and the values were normalized, this because, depending on the image, there may be more or fewer pixels, it will not always be the same; This ensures that, regardless of the number of pixels that exist, there are "relative" values.
    </li>
    <li>
        <b>E. Obtaining suitable components</b><br>
        In this part, those images that are part of the plate serial number will be extracted with the help of points of interest. The points of interest will be subjected to a hierarchical grouping using the "Ward" variance minimization algorithm, and thanks to the fact that the values of the points of interest are normalized, we can determine that these points will always vary by a distance between 0 and 1. , where 0.15 is the appropriate distance where usually all the members belonging to the serial code of the board form a single cluster together.<br><br>
        <div align="center">
            <img width="60%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/methodology/methodology_5.png?raw=true" />
            <p>Hierarchical clustering dendrogram showing that all points of interest in the rectangles of each image, in this example, lie within the distance of 0.15.</p>
        </div><br>
        The next thing is to find the cluster that contains the largest number of elements, since this is the one that ensures that all its points of interest are relatively similar and because normally a license plate usually has between 4 and 9 characters, and with all the restrictions placed previously, the remaining images and rectangles have decreased too much, for this reason, the group that turns out to contain the greatest number of elements at this separation distance, has the highest probability of being the correct one, since it is too much of a coincidence than at a distance less than or equal to 0.15, there are very similar elements that form a large group and that are not the characters that make up the plate serial code.<br>
        Once the largest cluster is obtained, we choose the images belonging to said cluster and discard those that are not part of it; In this way, we can say that we already have, to a certain extent, the appropriate components.<br><br>
        <div align="center">
            <img width="50%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/methodology/methodology_6.png?raw=true" />
            <p>Images within the largest cluster that belong to the vehicle's license plate serial code.</p>
        </div><br>
    </li>
    <li>
        <b>F. Image ordering</b><br>
        The order of the images that we have does not correspond to the actual order in which they are found on the plate, they are usually randomized, therefore, to find the appropriate order, we order the images from smallest to largest according to their 'x' coordinate of the point of origin (vertex of the upper left side of the rectangle of the image), in this simple way, we managed to put the images in the correct order.<br><br>
        <div align="center">
            <img width="50%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/methodology/methodology_7.png?raw=true" />
            <p>Images ordered correctly.</p>
        </div><br>
    </li>
    <li>
        <b>G. Component Image Processing</b><br>
        Again, image processing must be applied to each image of the obtained components (which contain a character from the board). The previously used image filters are applied: conversion to grayscale; Gaussian blur; and here the difference is that we create two images based on it: to one we apply Otsu's binary thresholding, and to another the same, but being the inverse binarization; this was intended to cope with whatever color type the plate uses. In later processes it will be decided which binarization is correct.<br>
        These two new images created based on each image that is contained, should be saved, since they are useful for the next step.
    </li>
    <li>
        <b>H. Character predictions based on probability</b><br>
        Now, we will proceed to predict which character is most likely to be the one that contains the image. This process must be carried out for each of the two images obtained for each image of the components, that is, it must be applied separately for the binarization images and for the inverse binarization images.<br>
        First, the image matrix must be resized in a single dimension (thinning), and using the two models we have for the prediction of letters and numbers, we input said linear vector to both models to obtain their predictions, both in letters and numbers. in numbers. From the probability results in both cases, the value of the highest probability produced by each model is obtained, and depending on which probability is greater, it will be discerned whether it is a letter or a number. But in the event that the probability that it is a number is greater than the probability that it is a letter, it should be taken into account that the probability that it is a letter is not greater than or equal to 97%, since if it is it is, it means that it has more possibilities of being a letter, since the model that was trained exclusively with letters contains 26 classes (letters of the English alphabet) against the 10 classes that were used in the number model (numbers of the alphabet). 0 to 9), so if the probability of it being a letter is greater than or equal to 97%, it's too much of a coincidence that you've returned such a high probability having so many classes in your vector array. And if, otherwise, it is not true that the probability that it is a letter exceeds or is equal to 97%, then it will definitely be taken that it is a number.<br>
        In this way, it was obtained which character is suitable for both the binarized image and the inversely binarized image.
    </li>
    <li>
        <b>I. Decide the correct binarization</b><br>
        In this section, it will be decided which of the two types of binarization is the ideal one.<br>
        Up to this point, there should be 2N number of images, where N is the number of original images that make up the serial code of the board; N are binarized, and the other N of 2N are inversely binarized. A summation of the probabilities of each image must be made and check which summation is greater; the sum that turns out to be greater should be the correct binarization method, since it was the closest to having accurate predictions, contrary to the other one that, having the values inverted, tended to predict non-existent characters and hence its low probabilities.<br>
        Once the proper binarization has been determined, we discard the other images and data, leaving only the ones that should be.
    </li>
    <li>
        <b>J. Discard images for low probability</b><br>
        Once the correct characters of the images belonging to the serial code of the car plate are obtained, a check must be made that the probability that it is said character is greater than or equal to 40%, this for each character. 40% was chosen to give a considerable margin of error, since we want the model to be robust, and, in these models, having less than 50% of that character does not mean that it is a bad prediction, more well indicates that it may be that character because there are enough matches, but it is not entirely clear.<br>
        Those characters that do not meet 40% of being the character they claim to be will be discarded, since it is probably an intrusive image that met the other requirements of the restrictions that were imposed.<br>
        So far, you should have what are possibly the correct characters that are in the serial code of the board.
    </li>
    <li>
        <b>K. Form the chain</b><br>
        Finally, it is enough to concatenate the resulting characters and deliver a single "string" type variable to the plate detector so that it can place it as the plate identifier.<br><br>
        <div align="center">
            <img width="50%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/methodology/methodology_8.png?raw=true" />
            <p>Completed plate prediction.</p>
        </div><br>
    </li>
</ul>

<h2><b>4. Results</b></h2>

As previously mentioned, the results of the models used for the detection of letters and numbers were very good, having, in the case of the model for letters, an accuracy of 0.908 and an R2 of 0.844; and, in the case of the number model, an accuracy of 0.949 and an R2 of 0.872.<br>
The results delivered by the proposed methodology, in general, were very good, being able to read plates that have a weak contrast or that have different colors, it is even capable of doing its work on plates with a bit of unnecessary saturation.<br>
In general, from an exclusive dataset to verify the effectiveness of the algorithm, made up of 148 plates from different countries, types, colors, fonts and text locations, it managed to hit 80 plates perfectly (without any type of error), others 43 plates had only one error and the remaining ones had more than one error. To observe the number of errors, the Levenshtein distance algorithm was used to compare two strings. The standard deviation of 1.312 indicates that there is usually an error between predictions, and the average distance gives us 0.831, indicating the same as the standard deviation, that it is possible that there may be an error for each prediction between 4-9 characters. who usually owns a license plate.<br>
Obviously, the best results were given in boards of the European, Japanese, Chinese, Argentinian, Russian, etc. type, because these boards have little information saturation and, normally, the contrast of the components is very high. In the case of license plates in the United States, the results were mixed, since license plates in this country can be customized and tend to have an excessive saturation of components and colors, making it difficult to recognize the serial code. But, even if they are diverse, as long as the letters can be discerned, the information can be read correctly.<br>
Therefore, these success rates could be increased if only plates from China, Russia, Japan, etc. had been placed, and, conversely, decreased if only complex plates had been placed. For this reason, various plates were put on it, to avoid any type of "favoritism" towards a type of plate.<br>
However, the methodology presents some confusion between similar characters, hence the problem that arose in most of the 43 plates that presented a single error. The characters that are often confused are the following: between 'G' and '6'; between '0' and '0'; between 'B' and '8'; between 'D' and 'O'; between 'I' and '1'; between 'Z' and '2'; etc. This can reduce the accuracy of the model, but the confusion is understandable, since at the time of image processing some samples may have remained similar and hence the errors between these characters with similarities. In addition to that, in themselves, these pairs of characters tend to be very similar.<br><br>

<div align="center">
    <img width="40%" src="https://github.com/ASASauqui/Car-license-plate-detector-and-reader/blob/main/Readme%20Images/methodology/methodology_9.png?raw=true" />
    <p>Some predictions.</p>
</div>
