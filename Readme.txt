This folder contains 2 folders and one python file and 2 text files except readme file
1- Evaluation (It has files which contain Result files w.r.t preprocessing filter size like 0by0,3by3,5by5,10by10)
2- Results folder (It contains Images for each step performed in the proposed methodology like classification, abundance map, postprocessing)
3- link.txt (it contains list of Hyperspectral files path that is used in process.py)
4- truth.txt (it contains list of ground truth files path that is used in process.py)
5- txt.txt (it contain images names which will be used to create folder/file names in Result folders)
6- process.py (It is the main file in which all functions and code for proposed methodology is presented. It is executed by typing command "python process.py")
7- install.txt (It contains pip install commands)

Steps to Run process.py
1- First Install Prerequisite packages in file names as "install.txt"
2- write & verify path of ground truth images in truth.txt file
3- write & verify path of HSI images in link.txt
4- write & verify labels of HSI images in txt.txt
5- execute process.py using command "python process.py"
