import cv2
import numpy as np
import sys

st = ''
ori = ''
matrix = np.zeros((9,9),np.int8)

def get_biggest_contour(image):
    biggest = None
    max_area = 0
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for j in range(0,len(contours)):
            i = contours[j].astype('int')
            area = cv2.contourArea(i)
            if area > 100 and area > max_area:
                    peri = cv2.arcLength(i,True)
                    approx = cv2.approxPolyDP(i,0.02*peri,True)
                    biggest = approx
                    max_area = area
    return biggest

def get_img():
    camera = cv2.VideoCapture(0)
    while True:
        img = camera.read()
        thresh = cv2.cvtColor(img[1],cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(thresh,255,1,1,11,2)
        thresh = get_biggest_contour(thresh)                    
        cv2.drawContours(img[1],[thresh.astype('int')],0,(0,255,0),3)
        cv2.imshow("Sudoku Solver",img[1])
        if (cv2.waitKey(5) != -1):           
            return [thresh,img[1]]
            break

def warp_img(contour,image):
    contour = contour.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)
    add = contour.sum(1)
    hnew[0] = contour[np.argmin(add)]
    hnew[2] = contour[np.argmax(add)]
    diff = np.diff(contour,axis = 1)
    hnew[1] = contour[np.argmin(diff)]
    hnew[3] = contour[np.argmax(diff)]
    h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
    retval = cv2.getPerspectiveTransform(hnew,h)
    image = cv2.warpPerspective(image,retval,(450,450))
    cv2.imshow("Sudoku Solver",image)
    return image

def ocr_read():
    img_rgb = cv2.imread('warped.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray,(5,5),1)
    for i in range(1,10):
        template = cv2.imread('ocr_template/'+str(i)+'.png',0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= 0.8)
        for pt in zip(*loc[::-1]):
            matrix[pt[1]/50][pt[0]/50] = i
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
    cv2.imshow('Sudoku Solver',img_rgb)
    cv2.waitKey()   

def same_row(i,j): return (i/9 == j/9)                          #SUDOKU Block
def same_col(i,j): return (i-j) % 9 == 0
def same_block(i,j): return (i/27 == j/27 and i%9/3 == j%9/3)
def r(a):
    i = a.find('0')
    if i == -1:
        show_ans(a)

    excluded_numbers = set()
    for j in range(81):
        if same_row(i,j) or same_col(i,j) or same_block(i,j):
            excluded_numbers.add(a[j])

    for m in '123456789':
        if m not in excluded_numbers:
            r(a[:i]+m+a[i+1:])

def rectify(h):
        h = h.reshape((4,2))
        hnew = np.zeros((4,2),dtype = np.float32)
        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]
        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]
        return hnew

def show_ans(stt):
    w_overlay = np.zeros((480,640,3), np.uint8)
    overlay = np.zeros((480,640,3), np.uint8)
    for i in range(0,81):
        if ori[i] == '0':
            cv2.putText(overlay,stt[i],((i%9)*71 + 20 ,(i/9)*53 + 40), cv2.FONT_HERSHEY_PLAIN, 3.0,(0,255,0),3)
    
    camera = cv2.VideoCapture(0)
    while True:
        img = camera.read()
        thresh = cv2.cvtColor(img[1],cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(thresh,255,1,1,11,2)    
        thresh = get_biggest_contour(thresh)
        if(len(thresh) == 4):
            thresh = rectify(thresh)
            h = np.array([ [0,0],[640,0],[640,480],[0,480] ],np.float32)
            retval = cv2.getPerspectiveTransform(h,thresh)
            w_overlay = cv2.warpPerspective(overlay,retval,(640,480)) 
       
        cv2.drawContours(img[1],[thresh.astype('int')],0,(0,0,255),3)
        cv2.imshow("Sudoku Solver",cv2.addWeighted(img[1],1,w_overlay,1,1))
        if (cv2.waitKey(5) != -1):            
            sys.exit()                   

img = get_img();
cv2.imshow("Sudoku Solver",img[1])
img = warp_img(img[0], img[1])

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img,(15,15),1)
cv2.imwrite("warped.jpg",cv2.adaptiveThreshold(img,255,1,1,11,2))
cv2.waitKey()

ocr_read()

for i in matrix:
    for j in i:
        ori+=str(j)
cv2.imshow("Sudoku Solver",cv2.imread("wait.jpg"))
cv2.waitKey()
print 'started'
r(ori)