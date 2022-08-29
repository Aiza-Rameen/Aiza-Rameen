import os
import numpy as np                          
import cv2                             
import matplotlib.pyplot as plt
from glob import glob                       
from osgeo import gdal
from osgeo import gdalconst
from spectral import *
import pysptools.eea as eea
import pysptools.abundance_maps as amp
import pysptools.distance as dst
import pysptools.classification as cls
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.metrics import structural_similarity as ssim


# For reading text files containing links to HSI Images (link.txt), Labels for Images(txt.txt) and Ground Truth Images (truth.txt)
file1 = open('link.txt', 'r')
Lines = file1.readlines()
for i in range(len(Lines)):
    Lines[i] = Lines[i].rstrip("\n")
file2 = open('txt.txt', 'r')
Labels = file2.readlines()
for i in range(len(Labels)):
    Labels[i] = Labels[i].rstrip("\n")
links1 = open('truth.txt', 'r')
Truths = links1.readlines()
for i in range(len(Truths)):
    Truths[i] = Truths[i].rstrip("\n")

# class for reading HSI Image (.bil) file and returning its data in array
class BilFile(object):

    def __init__(self, bil_file):
        self.bil_file = bil_file
        self.hdr_file = bil_file.split('.')[0]+'.hdr'

    def get_array(self, mask=None):
        self.nodatavalue, self.data = None, []
        gdal.GetDriverByName('EHdr').Register()
        img = gdal.Open(self.bil_file, gdalconst.GA_ReadOnly)
        for i in range(img.RasterCount):
            dat = None
            band = img.GetRasterBand((i+1))
            self.nodatavalue = band.GetNoDataValue()
            self.ncol = img.RasterXSize
            self.nrow = img.RasterYSize
            geotransform = img.GetGeoTransform()
            self.originX = geotransform[0]
            self.originY = geotransform[3]
            self.pixelWidth = geotransform[1]
            self.pixelHeight = geotransform[5]
            dat = band.ReadAsArray()
            dat = np.ma.masked_where(dat==self.nodatavalue, dat)
            #if mask is not None:
            #    dat = np.ma.masked_where(mask==True, dat)
            self.data.append(dat);    
        return self.nodatavalue, self.data

#method for SAM Classifier
def SAMs(data,E,thrs=None):
    clas = cls.SAM()
    cmap = clas.classify(data,E,thrs)
    clas = None
    return cmap
#method for SID Classifier
def SIDs(data,E,thrs=None):
    clas = cls.SID()
    cmap = clas.classify(data,E,thrs)
    clas = None
    return cmap

#method for UCLS Abundance Map
def apply_ucls(M,U):
    obj = amp.UCLS()
    res = obj.map(M,U)
    obj = None
    return res

#method for NNLS Abundance Map    
def apply_nnls(M,U):
    obj = amp.NNLS()
    res = obj.map(M,U)
    obj = None
    return res

#method for FCLS Abundance Map 
def apply_fcls(M,U):
    print(str(M.shape)+" : "+str(U.shape))
    obj = amp.FCLS()
    res = obj.map(M,U)
    obj = None
    return res

#method load HSI Image and apply Preprocessing 
def load_bill_file(input_url,parent_dir,text,blur_filter_size):
    #input_url = 'I:/HSI_OverlappingSigns/25Bg/2.bil'
    
    BillObj = BilFile(input_url)
    res = BillObj.get_array()
    BillObj = None
    n = []
    count = 0
    for i in res[1]:
        if count > 15 and count < 201:
            ind = cv2.blur(i,(blur_filter_size,blur_filter_size))
            n.append(ind)
        count = count + 1
    return np.asarray(np.dstack(n))


#method for ATGP Endmember Extraction
def apply_atgp(input_array,no_of_endmembers):
    obj = eea.ATGP()
    U = obj.extract(input_array,no_of_endmembers)
    obj = None
    return U
#method for FIPPI Endmember Extraction
def apply_fippi(input_array,no_of_endmembers):
    obj = eea.FIPPI()
    U = obj.extract(input_array,no_of_endmembers)
    obj = None
    return U
#method for NFINDR Endmember Extraction
def apply_nfindr(input_array,no_of_endmembers):
    obj = eea.NFINDR()
    U = obj.extract(input_array,no_of_endmembers)
    obj = None
    return U

#Post Processing Algorithm
def postprocessing(img_url):
    # the parameters are used to remove small size connected pixels outliar 
    constant_parameter_1 = 200
    constant_parameter_2 = 250
    constant_parameter_3 = 180

    # the parameter is used to remove big size connected pixels outliar
    constant_parameter_4 = 10

    # read the input image
    img = cv2.imread(img_url, 0)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

    # connected component analysis by scikit-learn framework
    #print("mean is: "+str(img.mean()))
    #print(img)
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    image_label_overlay = label2rgb(blobs_labels, image=img)

    fig, ax = plt.subplots(figsize=(10, 10))

    ...
    # plot the connected components (for debugging)
    #ax.imshow(image_label_overlay)
    #ax.set_axis_off()
    #plt.tight_layout()
    #.show()
    ...

    the_biggest_component = 0
    total_area = 0
    counter = 0
    average = 0.0
    area_array = []
    for region in regionprops(blobs_labels):
        area_array.append(region.area)
        if (region.area > 10):
            total_area = total_area + region.area
            counter = counter + 1
        # print region.area # (for debugging)
        # take regions with large enough areas
        if (region.area >= 250):
            if (region.area > the_biggest_component):
                the_biggest_component = region.area

    if(counter != 0):
        average = (total_area/counter)
    else:
        average = 0
    #print("the_biggest_component: " + str(the_biggest_component))
    #print("average: " + str(average))
    #print("number of regions: "+str(counter))
    #print("region areas: "+str(area_array))

    # experimental-based ratio calculation, modify it for your cases
    # a4_small_size_outliar_constant is used as a threshold value to remove connected outliar connected pixels
    # are smaller than a4_small_size_outliar_constant for A4 size scanned documents
    a4_small_size_outliar_constant = ((average/constant_parameter_1)*constant_parameter_2)+constant_parameter_3
    #print("a4_small_size_outliar_constant: " + str(a4_small_size_outliar_constant))

    # experimental-based ratio calculation, modify it for your cases
    # a4_big_size_outliar_constant is used as a threshold value to remove outliar connected pixels
    # are bigger than a4_big_size_outliar_constant for A4 size scanned documents
    a4_big_size_outliar_constant = a4_small_size_outliar_constant*constant_parameter_4
    #print("a4_big_size_outliar_constant: " + str(a4_big_size_outliar_constant))

    # remove the connected pixels are smaller than a4_small_size_outliar_constant
    pre_version = morphology.remove_small_objects(blobs_labels, average)
    # remove the connected pixels are bigger than threshold a4_big_size_outliar_constant 
    # to get rid of undesired connected pixels such as table headers and etc.
    component_sizes = np.bincount(pre_version.ravel())
    #print("component sizes: "+str(component_sizes))
    #print("a4_big_size_outliar_constant: "+str(a4_big_size_outliar_constant))
    #print("pre version shape: "+str(blobs_labels.shape))

    #too_small = component_sizes > (a4_big_size_outliar_constant)
    #too_small_mask = too_small[pre_version]
    #pre_version[too_small_mask] = 0

    # save the the pre-version which is the image is labelled with colors
    # as considering connected components
    #plt.imsave('pre_version.png', pre_version)
    #cv2.imwrite("pre_output.png", pre_version)

    # read the pre-version
    #img = cv2.imread('pre_version.png', 0)
    #print("preversion: "+str(pre_version))
    for x in range(pre_version.shape[0]):
        for y in range(pre_version.shape[1]):
            if pre_version[x][y] > 0:
                pre_version[x][y] = 0
            else:
                pre_version[x][y] = 255

    # ensure binary
    #img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # save the the result
    cv2.imwrite(img_url+"_postprocessed.png", pre_version)
    return pre_version

#convert Array to Binary Image for Classification
def convert_array(inp):
    #print(inp.shape)
    for j in range(800):
        for k in range(640):
            if inp[j][k] < 240:
                inp[j][k] = 0
            else:
                inp[j][k] = 255
    return inp

#Method for calculating Accuracy
def calculate_accuracy(inp,gt):
    total = 0
    connfirm = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    f1 = 0
    for j in range(800):
        for k in range(640):
            if inp[j][k] == 0 and gt[j][k] == 0:
                true_positive = true_positive + 1
            if inp[j][k] == 0 and gt[j][k] > 0:
                false_positive = false_positive + 1
            if inp[j][k] > 0 and gt[j][k] > 0:
                true_negative = true_negative + 1
            if inp[j][k] > 0 and gt[j][k] == 0:
                false_negative = false_negative + 1
            total = total + 1
    #accuracy = round(((correct/total)*100),2)
    dom = true_positive+true_negative+false_positive+false_negative
    num = true_positive+true_negative
    dom1 = true_positive+false_positive
    dom2 = true_positive+false_negative
    accuracy = round((num/dom),4)
    precision = round((true_positive/dom1),4)
    recall = round((true_positive/dom2),4)
    c = precision*recall
    d = precision+recall
    if d!= 0 :
        f1 = round((c/d),4) + round((c/d),4)
    return "Accuracy: "+str(accuracy)+", Precision: "+str(precision)+", Recall: "+str(recall)+", F1 Score: "+str(f1)+", True Positive: "+str(true_positive)+" , False Positive: "+str(false_positive)+" , True Negative: "+str(true_negative)+" , False Negative: "+str(false_negative)

#Method for calculating Distance like SAM, SID
def calculate_distance(inp,gt):
    inp1 = inp.flatten()
    gt1 = gt.flatten()
    distance = dst.NormXCorr(inp1,gt1)
    distance1 = dst.chebyshev(inp1,gt1)
    distance2 = dst.SAM(inp1,gt1)
    distance3 = dst.SID(inp1,gt1)
    if distance is not None:
        distance = round(distance,5)
    else:
        distance = -1

    if distance1 is not None:
        distance1 = round(distance1,5)
    else:
        distance1 = -1
    
    if distance2 is not None:
        distance2 = round(distance2,5)
    else:
        distance2 = -1
    
    if distance3 is not None:
        distance3 = round(distance3,5)
    else:
        distance3 = -1
    return ", NormXCorr Distance ,"+str(distance)+", Chebychev Distance," + str(distance1)+", SAM Distance," + str(distance2)+", SID Distance," + str(distance3)

#method to apply post processing and evaluation with ground truth
def evalutate(url,url1):
    gtruth = convert_array(cv2.imread(url1,0))
    for img in glob(url):
        n= postprocessing(img)
        n = convert_array(n)
        name = os.path.basename(img)
        txt = name+ " : " + str(calculate_accuracy(n,gtruth))
        #print(str(values))
        with open("evaluation.txt", "a") as txt_file:
            txt_file.write(txt + "\n")
    return True
#method to Save Abundance Maps
def save_abundanceMaps(maps_array,parent_dir,text,method,map_name):
    newFi = np.dsplit(maps_array,maps_array.shape[2])
    newFi1= [[[0 for y in range(640)] for y in range(800)] for x in range(maps_array.shape[2])]
    for i in range(maps_array.shape[2]):
        for j in range(800):
            for k in range(640):
                newFi1[i][j][k] = float(newFi[i][j][k][0])
    final = np.array(newFi1,copy=True)
    OldMax = np.amax(final)
    OldMin = np.amin(final)
    NewMax = 255
    NewMin = 0
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)
    path = os.path.join(parent_dir, "AbundanceMap")
    isExist = os.path.exists(path)
    if not isExist:
        os.mkdir(path)
    for i in range(final.shape[0]):
        for j in range(800):
            for k in range(640):
                final[i][j][k] = ((((final[i][j][k] - OldMin) * NewRange) / OldRange) + NewMin)
    for i in range(final.shape[0]):
        cv2.imwrite(os.path.join(path , method+'_'+map_name+'_abundance_img_'+text+'_'+str(i)+'.jpg'), final[i])

#method to Check Existing Abundance Maps
def check_abundanceMaps(rows,parent_dir,text,method,map_name):
    path = os.path.join(parent_dir, "AbundanceMap")
    isExists = os.path.exists(path)
    if not isExists:
        os.mkdir(path)
    isExist = False
    for i in range(rows):
        a = os.path.exists(os.path.join(path , method+'_'+map_name+'_abundance_img_'+text+'_'+str(i)+'.jpg'))
        if a == True or isExist == True:
            isExist = True
        else:
            isExist = False
    return isExist

#method to calculate SAM Sid Distance and Compare it with Abundance Maps 
def save_sam_sid_results(y,res_fcls,res_ucls,res_nnls,cmap1,parent_dir,text,method):
    endm1fcls = []
    endm2fcls = []
    endm3fcls = []
    endm1ucls = []
    endm2ucls = []
    endm3ucls = []
    endm1nnls = []
    endm2nnls = []
    endm3nnls = []
    y1 = []
    y30 = []
    y60 = []
    y90 = []
    y120 = []
    y150 = []
    y180 = []
    main_fcls = np.array(res_fcls,copy=True)
    main_ucls = np.array(res_ucls,copy=True)
    main_nnls = np.array(res_nnls,copy=True)
    for i in range(800):
        for j in range(640):
            if cmap1[i][j] == 0:
                endm1fcls.append(main_fcls[i][j][0])
                endm2fcls.append(main_fcls[i][j][1])
                if len(main_fcls[i][j]) > 2:
                    endm3fcls.append(main_fcls[i][j][2])
                else:
                    endm3fcls.append(0)
                
                endm1ucls.append(main_ucls[i][j][0])
                endm2ucls.append(main_ucls[i][j][1])
                if len(main_ucls[i][j]) > 2:
                    endm3ucls.append(main_ucls[i][j][2])
                else:
                    endm3ucls.append(0)
                
                endm1nnls.append(main_nnls[i][j][0])
                endm2nnls.append(main_nnls[i][j][1])
                if len(main_nnls[i][j]) > 2:
                    endm3nnls.append(main_nnls[i][j][2])
                else:
                    endm3nnls.append(0)
                y1.append(y[i][j][0])
                y30.append(y[i][j][29])
                y60.append(y[i][j][59])
                y90.append(y[i][j][89])
                y120.append(y[i][j][119])
                y150.append(y[i][j][149])
                y180.append(y[i][j][179])
    path = os.path.join(parent_dir, text)
    isExist = os.path.exists(path)
    if not isExist:
        os.mkdir(path)
    f = open(os.path.join(path , method+'_results.txt'), "a")
    print('SID\n',file=f)

    print('fcls Channel 16, endmember1: '+str(round(dst.SID(y1,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 16, endmember2: '+str(round(dst.SID(y1,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 16, endmember3: '+str(round(dst.SID(y1,endm3fcls),2))+' \n',file=f)
    
    print('fcls Channel 46, endmember1: '+str(round(dst.SID(y30,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 46, endmember2: '+str(round(dst.SID(y30,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 46, endmember3: '+str(round(dst.SID(y30,endm3fcls),2))+' \n',file=f)
    
    print('fcls Channel 76, endmember1: '+str(round(dst.SID(y60,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 76, endmember2: '+str(round(dst.SID(y60,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 76, endmember3: '+str(round(dst.SID(y60,endm3fcls),2))+' \n',file=f)
    
    print('fcls Channel 106, endmember1: '+str(round(dst.SID(y90,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 106, endmember2: '+str(round(dst.SID(y90,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 106, endmember3: '+str(round(dst.SID(y90,endm3fcls),2))+' \n',file=f)
    
    print('fcls Channel 136, endmember1: '+str(round(dst.SID(y120,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 136, endmember2: '+str(round(dst.SID(y120,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 136, endmember3: '+str(round(dst.SID(y120,endm3fcls),2))+' \n',file=f)
    
    print('fcls Channel 166, endmember1: '+str(round(dst.SID(y150,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 166, endmember2: '+str(round(dst.SID(y150,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 166, endmember3: '+str(round(dst.SID(y150,endm3fcls),2))+' \n',file=f)
    
    print('fcls Channel 196, endmember1: '+str(round(dst.SID(y180,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 196, endmember2: '+str(round(dst.SID(y180,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 196, endmember3: '+str(round(dst.SID(y180,endm3fcls),2))+' \n',file=f)

    #UCLS
    print('UCLS Channel 16, endmember1: '+str(round(dst.SID(y1,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 16, endmember2: '+str(round(dst.SID(y1,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 16, endmember3: '+str(round(dst.SID(y1,endm3ucls),2))+' \n',file=f)
    
    print('UCLS Channel 46, endmember1: '+str(round(dst.SID(y30,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 46, endmember2: '+str(round(dst.SID(y30,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 46, endmember3: '+str(round(dst.SID(y30,endm3ucls),2))+' \n',file=f)
    
    print('UCLS Channel 76, endmember1: '+str(round(dst.SID(y60,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 76, endmember2: '+str(round(dst.SID(y60,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 76, endmember3: '+str(round(dst.SID(y60,endm3ucls),2))+' \n',file=f)
    
    print('UCLS Channel 106, endmember1: '+str(round(dst.SID(y90,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 106, endmember2: '+str(round(dst.SID(y90,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 106, endmember3: '+str(round(dst.SID(y90,endm3ucls),2))+' \n',file=f)
    
    print('UCLS Channel 136, endmember1: '+str(round(dst.SID(y120,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 136, endmember2: '+str(round(dst.SID(y120,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 136, endmember3: '+str(round(dst.SID(y120,endm3ucls),2))+' \n',file=f)
    
    print('UCLS Channel 166, endmember1: '+str(round(dst.SID(y150,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 166, endmember2: '+str(round(dst.SID(y150,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 166, endmember3: '+str(round(dst.SID(y150,endm3ucls),2))+' \n',file=f)
    
    print('UCLS Channel 196, endmember1: '+str(round(dst.SID(y180,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 196, endmember2: '+str(round(dst.SID(y180,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 196, endmember3: '+str(round(dst.SID(y180,endm3ucls),2))+' \n',file=f)

    #NNLS
    print('NNLS Channel 16, endmember1: '+str(round(dst.SID(y1,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 16, endmember2: '+str(round(dst.SID(y1,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 16, endmember3: '+str(round(dst.SID(y1,endm3nnls),2))+' \n',file=f)
    
    print('NNLS Channel 46, endmember1: '+str(round(dst.SID(y30,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 46, endmember2: '+str(round(dst.SID(y30,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 46, endmember3: '+str(round(dst.SID(y30,endm3nnls),2))+' \n',file=f)
    
    print('NNLS Channel 76, endmember1: '+str(round(dst.SID(y60,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 76, endmember2: '+str(round(dst.SID(y60,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 76, endmember3: '+str(round(dst.SID(y60,endm3nnls),2))+' \n',file=f)
    
    print('NNLS Channel 106, endmember1: '+str(round(dst.SID(y90,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 106, endmember2: '+str(round(dst.SID(y90,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 106, endmember3: '+str(round(dst.SID(y90,endm3nnls),2))+' \n',file=f)
    
    print('NNLS Channel 136, endmember1: '+str(round(dst.SID(y120,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 136, endmember2: '+str(round(dst.SID(y120,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 136, endmember3: '+str(round(dst.SID(y120,endm3nnls),2))+' \n',file=f)
    
    print('NNLS Channel 166, endmember1: '+str(round(dst.SID(y150,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 166, endmember2: '+str(round(dst.SID(y150,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 166, endmember3: '+str(round(dst.SID(y150,endm3nnls),2))+' \n',file=f)
    
    print('NNLS Channel 196, endmember1: '+str(round(dst.SID(y180,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 196, endmember2: '+str(round(dst.SID(y180,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 196, endmember3: '+str(round(dst.SID(y180,endm3nnls),2))+' \n',file=f)

    #SAM
    print('SAM\n',file=f)
    #fcls
    print('fcls Channel 16, endmember1: '+str(round(dst.SAM(y1,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 16, endmember2: '+str(round(dst.SAM(y1,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 16, endmember3: '+str(round(dst.SAM(y1,endm3fcls),2))+' \n',file=f)
    
    print('fcls Channel 46, endmember1: '+str(round(dst.SAM(y30,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 46, endmember2: '+str(round(dst.SAM(y30,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 46, endmember3: '+str(round(dst.SAM(y30,endm3fcls),2))+' \n',file=f)
    
    print('fcls Channel 76, endmember1: '+str(round(dst.SAM(y60,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 76, endmember2: '+str(round(dst.SAM(y60,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 76, endmember3: '+str(round(dst.SAM(y60,endm3fcls),2))+' \n',file=f)
    
    print('fcls Channel 106, endmember1: '+str(round(dst.SAM(y90,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 106, endmember2: '+str(round(dst.SAM(y90,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 106, endmember3: '+str(round(dst.SAM(y90,endm3fcls),2))+' \n',file=f)
    
    print('fcls Channel 136, endmember1: '+str(round(dst.SAM(y120,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 136, endmember2: '+str(round(dst.SAM(y120,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 136, endmember3: '+str(round(dst.SAM(y120,endm3fcls),2))+' \n',file=f)
    
    print('fcls Channel 166, endmember1: '+str(round(dst.SAM(y150,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 166, endmember2: '+str(round(dst.SAM(y150,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 166, endmember3: '+str(round(dst.SAM(y150,endm3fcls),2))+' \n',file=f)
    
    print('fcls Channel 196, endmember1: '+str(round(dst.SAM(y180,endm1fcls),2))+' \n',file=f)
    print('fcls Channel 196, endmember2: '+str(round(dst.SAM(y180,endm2fcls),2))+' \n',file=f)
    print('fcls Channel 196, endmember3: '+str(round(dst.SAM(y180,endm3fcls),2))+' \n',file=f)

    #UCLS
    print('UCLS Channel 16, endmember1: '+str(round(dst.SAM(y1,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 16, endmember2: '+str(round(dst.SAM(y1,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 16, endmember3: '+str(round(dst.SAM(y1,endm3ucls),2))+' \n',file=f)
    
    print('UCLS Channel 46, endmember1: '+str(round(dst.SAM(y30,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 46, endmember2: '+str(round(dst.SAM(y30,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 46, endmember3: '+str(round(dst.SAM(y30,endm3ucls),2))+' \n',file=f)
    
    print('UCLS Channel 76, endmember1: '+str(round(dst.SAM(y60,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 76, endmember2: '+str(round(dst.SAM(y60,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 76, endmember3: '+str(round(dst.SAM(y60,endm3ucls),2))+' \n',file=f)
    
    print('UCLS Channel 106, endmember1: '+str(round(dst.SAM(y90,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 106, endmember2: '+str(round(dst.SAM(y90,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 106, endmember3: '+str(round(dst.SAM(y90,endm3ucls),2))+' \n',file=f)
    
    print('UCLS Channel 136, endmember1: '+str(round(dst.SAM(y120,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 136, endmember2: '+str(round(dst.SAM(y120,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 136, endmember3: '+str(round(dst.SAM(y120,endm3ucls),2))+' \n',file=f)
    
    print('UCLS Channel 166, endmember1: '+str(round(dst.SAM(y150,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 166, endmember2: '+str(round(dst.SAM(y150,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 166, endmember3: '+str(round(dst.SAM(y150,endm3ucls),2))+' \n',file=f)
    
    print('UCLS Channel 196, endmember1: '+str(round(dst.SAM(y180,endm1ucls),2))+' \n',file=f)
    print('UCLS Channel 196, endmember2: '+str(round(dst.SAM(y180,endm2ucls),2))+' \n',file=f)
    print('UCLS Channel 196, endmember3: '+str(round(dst.SAM(y180,endm3ucls),2))+' \n',file=f)

    #NNLS
    print('NNLS Channel 16, endmember1: '+str(round(dst.SAM(y1,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 16, endmember2: '+str(round(dst.SAM(y1,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 16, endmember3: '+str(round(dst.SAM(y1,endm3nnls),2))+' \n',file=f)
    
    print('NNLS Channel 46, endmember1: '+str(round(dst.SAM(y30,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 46, endmember2: '+str(round(dst.SAM(y30,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 46, endmember3: '+str(round(dst.SAM(y30,endm3nnls),2))+' \n',file=f)
    
    print('NNLS Channel 76, endmember1: '+str(round(dst.SAM(y60,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 76, endmember2: '+str(round(dst.SAM(y60,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 76, endmember3: '+str(round(dst.SAM(y60,endm3nnls),2))+' \n',file=f)
    
    print('NNLS Channel 106, endmember1: '+str(round(dst.SAM(y90,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 106, endmember2: '+str(round(dst.SAM(y90,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 106, endmember3: '+str(round(dst.SAM(y90,endm3nnls),2))+' \n',file=f)
    
    print('NNLS Channel 136, endmember1: '+str(round(dst.SAM(y120,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 136, endmember2: '+str(round(dst.SAM(y120,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 136, endmember3: '+str(round(dst.SAM(y120,endm3nnls),2))+' \n',file=f)
    
    print('NNLS Channel 166, endmember1: '+str(round(dst.SAM(y150,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 166, endmember2: '+str(round(dst.SAM(y150,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 166, endmember3: '+str(round(dst.SAM(y150,endm3nnls),2))+' \n',file=f)
    
    print('NNLS Channel 196, endmember1: '+str(round(dst.SAM(y180,endm1nnls),2))+' \n',file=f)
    print('NNLS Channel 196, endmember2: '+str(round(dst.SAM(y180,endm2nnls),2))+' \n',file=f)
    print('NNLS Channel 196, endmember3: '+str(round(dst.SAM(y180,endm3nnls),2))+' \n',file=f)

    f.close()

#method to Save SAM/SID Classifier Image with ATGP Endmember Extraction 
def save_images_classifier_atgp(parent_dir,text,array,method,map_name):
    path = os.path.join(parent_dir, "Classification")
    isExist = os.path.exists(path)
    if not isExist:
        os.mkdir(path)
    arr = np.array(array,copy=True)
    for j in range(800):
        for k in range(640):
            if array[j][k] < 1 or array[j][k] > 1:
                arr[j][k] = 0
            else:
                arr[j][k] = 255
    cv2.imwrite(os.path.join(path , method+'_'+map_name+'_classifier_'+text+'.jpg'), arr)
    return arr

#method to Save SAM/SID Classifier Image with FIPPI Endmember Extraction 
def save_images_classifier_fippi(parent_dir,text,array,method,map_name):
    path = os.path.join(parent_dir, "Classification")
    isExist = os.path.exists(path)
    if not isExist:
        os.mkdir(path)
    arr = np.array(array,copy=True)
    for j in range(800):
        for k in range(640):
            if array[j][k] < 1:
                arr[j][k] = 0
            else:
                arr[j][k] = 255
    cv2.imwrite(os.path.join(path , method+'_'+map_name+'_classifier_'+text+'.jpg'), arr)
    return arr

#method to Save SAM/SID Classifier Image with NFIDNR Endmember Extraction 
def save_images_classifier_nfindr(parent_dir,text,array,method,map_name):
    path = os.path.join(parent_dir, "Classification")
    isExist = os.path.exists(path)
    if not isExist:
        os.mkdir(path)
    arr = np.array(array,copy=True)
    for j in range(800):
        for k in range(640):
            if array[j][k] < 2:
                arr[j][k] = 0
            else:
                arr[j][k] = 255
    cv2.imwrite(os.path.join(path , method+'_'+map_name+'_classifier_'+text+'.jpg'), arr)
    return arr

#Main Method Complete with ATGP
def process_agtp(lblsi,parent_dir,y):
    print('Processing Image - '+lblsi+' ATGP')
    u = apply_atgp(y,3)
    if check_abundanceMaps(u.shape[0],parent_dir,lblsi,'ATGP','UCLS') == False:
        print('Processing Image - '+lblsi+' ATGP - UCLS')
        aban_maps = apply_ucls(y,u)
        save_abundanceMaps(aban_maps,parent_dir,lblsi,'ATGP','UCLS')
    if check_abundanceMaps(u.shape[0],parent_dir,lblsi,'ATGP','FCLS') == False:
        print('Processing Image - '+lblsi+' ATGP - FCLS')
        aban_maps1 = apply_fcls(y,u)
        save_abundanceMaps(aban_maps1,parent_dir,lblsi,'ATGP','FCLS')
    if check_abundanceMaps(u.shape[0],parent_dir,lblsi,'ATGP','NNLS') == False:
        print('Processing Image - '+lblsi+' ATGP - NNLS')
        aban_maps2 = apply_nnls(y,u)
        save_abundanceMaps(aban_maps2,parent_dir,lblsi,'ATGP','NNLS')
    print('Processing Image - '+lblsi+' SAM CLassifier')
    arry = SAMs(y,u,0.1)
    arry1 = SIDs(y,u,0.009)
    print('Saving Image - '+lblsi)
    arr = save_images_classifier_atgp(parent_dir,lblsi,arry,'ATGP','SAM')
    arr1 = save_images_classifier_atgp(parent_dir,lblsi,arry1,'ATGP','SID')
    #save_sam_sid_results(y,aban_maps1,aban_maps,aban_maps2,arr,parent_dir,lblsi,'ATGP')

#Main Method Complete with FIPPI
def process_fippi(lblsi,parent_dir,y):
    print('Processing Image - '+lblsi+' FIPPI')
    u = apply_fippi(y,3)
    print("fippi shape "+str(u.shape))
    if check_abundanceMaps(u.shape[0],parent_dir,lblsi,'FIPPI','UCLS') == False:
       ('Processing Image - '+lblsi+' FIPPI - UCLS')
       aban_maps = apply_ucls(y,u)
       save_abundanceMaps(aban_maps,parent_dir,lblsi,'FIPPI','UCLS')
    if check_abundanceMaps(u.shape[0],parent_dir,lblsi,'FIPPI','FCLS') == False:
       print('Processing Image - '+lblsi+' FIPPI - FCLS')
       aban_maps1 = apply_fcls(y,u)
       save_abundanceMaps(aban_maps1,parent_dir,lblsi,'FIPPI','FCLS')
    if check_abundanceMaps(u.shape[0],parent_dir,lblsi,'FIPPI','NNLS') == False:
        print('Processing Image - '+lblsi+' FIPPI - NNLS')
        aban_maps2 = apply_nnls(y,u)
        save_abundanceMaps(aban_maps2,parent_dir,lblsi,'FIPPI','NNLS')
    print('Processing Image - '+lblsi+' SAM CLassifier')
    arry = SAMs(y,u,0.1)
    arry1 = SIDs(y,u,0.009)
    print('Saving Image - '+lblsi)
    arr = save_images_classifier_fippi(parent_dir,lblsi,arry,'FIPPI','SAM')
    arr1 = save_images_classifier_fippi(parent_dir,lblsi,arry1,'FIPPI','SID')
    #save_sam_sid_results(y,aban_maps1,aban_maps,aban_maps2,arr,parent_dir,lblsi,'FIPPI')

#Main Method Complete with NFIDNR
def process_nfindr(lblsi,parent_dir,y):
    print('Processing Image - '+lblsi+' NFINDR')
    u = apply_nfindr(y,3)
    if check_abundanceMaps(u.shape[0],parent_dir,lblsi,'NFINDR','UCLS') == False:
        print('Processing Image - '+lblsi+' NFINDR - UCLS')
        aban_maps = apply_ucls(y,u)
        save_abundanceMaps(aban_maps,parent_dir,lblsi,'NFINDR','UCLS')
    if check_abundanceMaps(u.shape[0],parent_dir,lblsi,'NFINDR','FCLS') == False:
        print('Processing Image - '+lblsi+' NFINDR - FCLS')
        aban_maps1 = apply_fcls(y,u)
        save_abundanceMaps(aban_maps1,parent_dir,lblsi,'NFINDR','FCLS')
    if check_abundanceMaps(u.shape[0],parent_dir,lblsi,'NFINDR','NNLS') == False:
        print('Processing Image - '+lblsi+' NFINDR - NNLS')
        aban_maps2 = apply_nnls(y,u)
        save_abundanceMaps(aban_maps2,parent_dir,lblsi,'NFINDR','NNLS')
    print('Processing Image - '+lblsi+' SAM CLassifier')
    arry = SAMs(y,u,0.1)
    arry1 = SIDs(y,u,0.009)
    print('Saving Image - '+lblsi)
    arr = save_images_classifier_nfindr(parent_dir,lblsi,arry,'NFINDR','SAM')
    arr1 = save_images_classifier_nfindr(parent_dir,lblsi,arry1,'NFINDR','SID')
    #save_sam_sid_results(y,aban_maps1,aban_maps,aban_maps2,arr,parent_dir,lblsi,'NFINDR')

#method to create parent directory of Image for Storing Data in it
def create_parent_directory(lblsi,parent_dir):
    print('Creating Parent Directory - '+lblsi)
    path = os.path.join(parent_dir, lblsi)
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    return path
    

#main Loop for iteration of All HSI links and Images after reading from txt files
txts = Lines
lbls = Labels
folder_name = '5by5/' #number is used for separation of all results using 5 filter size in a separate folder
blur_filter_size = 5 #blur filter size
for i in range(len(txts)):
    parent_dir = 'D:/thesis/FinalProcess/Results/'+folder_name
    print('Loading Image - '+lbls[i])
    y = load_bill_file(txts[i],parent_dir,lbls[i],blur_filter_size) #load each line
    parent_dir = create_parent_directory(lbls[i],parent_dir)
    process_agtp(lbls[i],parent_dir,y)
    process_fippi(lbls[i],parent_dir,y)
    process_nfindr(lbls[i],parent_dir,y)
    y = None
    postp = parent_dir+'/Classification/*.jpg'
    evalutate(postp,Truths[i])
    
print('Processing Completed 100%')