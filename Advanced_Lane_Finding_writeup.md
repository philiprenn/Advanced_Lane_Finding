
# Self-Driving Car Engineer Nanodegree Writeup
***
## Project 2: **Advanced Lane Finding** 

#### **Written by: Philip Renn**

[image]: ./output_images/writeup_images/result.png
***
![alt text][image]
***

## Preliminary Goal
    Calibrate a 2D camera using chessboard calibration images to a aquire a conversion matrix and distortion coefficients.   

## Primary Goal
    Write a sofetware pipeline to identiify the lane boundaries from raw images from a video.

### Steps:
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## README
***
### Dependencies
***
   * Anaconda Prompt
   * Python 3
   * Matplotlib
   * Numpy
   * OpenCV
   * Math
   * Glob
   * Pickle
   * Moviepy
   * IPython
   * Collections
   * Jupyter Notebook

### Files
***
* `./camera_cal/`    | images used for calibrating the camera and the outputs of "camera_calibration.ipynb"
* `./test_images/`   | images to test pipeline
* `./output_images/` | processed test images of each step of pipeline

* `./Advanced_Lane_Finding.ipynb`   | main code for processing images and video
* `./project_video_output.mp4`      | video of processed "project_veideo.mp4"
* `./camera_calibration.ipynb`      | code to calibrate camera using images in ./camera_cal/
* `./calibration_data.p`            | file containing matrix and distortion coeficients

## How to run `Advanced_Lane_Finding.ipynb` using jupyter notebook
***
   1. Get repository from github
   2. Open an Anaconda Prompt
   3. cd into the directory containing the Advanced_Lane_Finding.ipynb file
   4. Enter the command: jupyter notebook P1.ipynb
   5. Click on the "Kernel" tab >> "Restart & Run All"
---

## Rubric Points
___
## Camera Calibration
### 1. Have the camera matrix and distortion coefficients been computed correctly and checked on the calibration test image?

The code for calibrating the camera is in the file `camera_calibration.ipynb`.

First, I read in all calibration images and prepared `objp` to store the object points (x,y,z) with a size to store coordinates in the wolrd, assuming z=0. This matrix is appended to `objpoints` for each detected image. Then, I detected the `corners` of the chessboard and appended these points to `imgpoints` for each detected chessboard. 

I then use `objpoints` and `imgpoints` to calibrate the camera using `cv2.calibrateCamera()` function, which will return `mtx` and `dist` which are the calibration matrix and distortion coefficients, respectively. The results of `mtx` and `dist` were exported to the file `calibration_data.p`. These are the variables that will be used to undistort the raw images from the video using `cv2.undistort()`. Below is an output of the calibration process: 

[image1]: ./output_images/writeup_images/cal_example.png
***
![alt text][image1]

***

## Pipeline (single images)
### 1. Has the distortion correction been correctly applied to each image?

Below is an example of the distortion correction step. I imported the `mtx` and `dist` data from the `calibration_data.p` file that was created in the camera calibration step. I used these values as arguments in the `cv2.undistort()` function which returned the undistorted image. The unditort process corrects for the distortion caused by the camera lens. A rounded camera lens captures more light at the edges of the image which causes objects to appear more or less curved than they actually are in the real world. This can be observed in the image below. Notice the difference in position of the white car from the original image and the undistorted image. In the orginal image the entire rear end of the white car is visible whereas the undistorted image show less of the rear because it was close to the edge of the frame which is generally where the "curved" effect can be observed.

[image2]: ./output_images/writeup_images/test1_undist.png
***
![alt_text][image2]
***


### 2. Has a perspective transform been applied to rectify the image?
Next, I applied the perspective transform using `M` to transform the selected region `src`, determined by the initialized points, to the destination `dst` region. This perspective transformation returns a "birds-eye" view. 
```python
#### Perspective Transform Variables ####
# Set offset for destination points
offsetx = 400
# Defining trapezoid points for src image 
imshape = (720, 1280)
left_bottom_p = [217, 705]
left_top_p = [588, 453]
right_top_p = [694, 453]
right_bottom_p = [1100, 705]
# Set src and dst points and get matrix (and inverse) to warp perspective
# Define 4 source points src = np.float32([[,],[,],[,],[,]])
src = np.float32([[left_bottom_p, left_top_p, right_top_p, right_bottom_p]], dtype=np.int32)
# Define 4 destination points dst = np.float32([[,],[,],[,],[,]])
dst = np.float32([[offsetx, imshape[0]],[offsetx, 0],[imshape[1]-offsetx, 0],[imshape[1]-offsetx, imshape[0]]])
# Use cv2.getPerspectiveTransform() to get M, the transform matrix, and its inverse, Minv
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
```

It is important to note that the lines should appear parallel in the transformed image since the lines appear straight in the undistorted image. The red lines represent the region specified by the source, `src`, and destination, `dst`, points. An example output of this step is shown below:

[image3]: ./output_images/writeup_images/straight_lines1_birdseye.png
***
![alt_text][image3]
***

### 3. Has a binary image been created using color transforms, gradients or other methods?
I created a binary image as a result of a combination of thresholding layers from difference color spaces. To do this, I transformed the BGR image to the HLS color space and thresholded each layer H,L,S. I also thresholded the red layer of the BGR image to detect the white lines as white is represented best in the red layer values. I used [colorizer](http://colorizer.org/) to tune the threshold values.

The code for the `color_select()` function below:

```python
def color_mask(img, img_hls):
    """
    Applies a color selection mask to find yellow & white lane line pixel values.
    Converts input image to HLS color space and thresholds each layer and performs
    bitwise operations with the Red layer from BGR image.
    
    Returns a color-masked binary image.
    """
    # Grab R layer of BGR image and H,L,S layers from HLS image
    R = img[:,:,2]
    H = img_hls[:,:,0]
    L = img_hls[:,:,1]
    S = img_hls[:,:,2]
    
    # Applying color thresholds using color space thresholds for R, H, L, S layers
    R_binary = np.zeros_like(R)
    R_binary[(R >= R_thresh[0]) & (L >= 150)] = 1
    H_binary = np.zeros_like(H)
    H_binary[(H >= H_thresh[0]) & (H <= H_thresh[1])] = 1
    L_binary = np.zeros_like(L)
    L_binary[(L >= L_thresh[0]) & (L <= L_thresh[1])] = 1
    S_binary = np.zeros_like(S)
    S_binary[(S >= S_thresh[0]) & (S <= S_thresh[1])] = 1
    
    # Combine hue and Lightness layers to detect lane lines in darker areas (e.g. shadows)
    HL_binary = cv2.bitwise_and(H_binary, L_binary)

    # Combine all thresholded layers
    mask_binary = np.zeros_like(R)
    mask_binary = cv2.bitwise_and(HL_binary, S_binary)
    mask_binary = cv2.bitwise_or(mask_binary, R_binary)    
    return mask_binary
```

[image4]: ./output_images/writeup_images/straight_lines1_red.png
[image5]: ./output_images/writeup_images/straight_lines1_hls.png
[image6]: ./output_images/writeup_images/straight_lines1_combined.png
[image7]: ./output_images/writeup_images/straight_lines1_gradient.png
[image8]: ./output_images/writeup_images/straight_lines1_roi.png
#### Red Layer
***
The values of the red layer are thresholded to select only the pixels with values above 200 and lightness value above 150, which will assign 1's to yellow and white values.
![alt_text][image4]

#### HLS Layers
***
I used the HLS color space to make my `color_select()` function more robust when conditions are not ideal (e.g. shadows, different road hue, brightness, etc.). The hue layer is does a good job of finding the base color independent of brightness which is helpful when shadows are present. I combined the lightness layer with the hue layer to find yellow and white values in dark areas. The saturation layer was used to verify the pixels detected in the low light areas are saturated with a high enough color value to be confident they are part of the lane line. 
![alt_text][image5]

#### Combined Color Spaces
***
The results of thresholding the HLS layers and the Red layer are combined using `cv2.bitwise_or()`. An example of the color selected image is displayed below:  
![alt_text][image6]

#### Gradient (Sobel)
***
Using the sobel function and a grayscaled copy of the image, I found the gradient of the x and y dimension and calculated the direction of the gradient. I thresholded the gradient in the x-dimension to find where there is a greater change in color intensity value. This is helpful when finding the side edges of lane lines because the lane lines generally contrast with the road which will result in a higher gradient value in the x-dimension. The direction will calculate the direction of the x and y gradient values. Essentially, I used the direction to be confident that the detected lane line pixels are more vertical than horizontal. The direction acts as a mask of the `color_select` image and `gradx`. Notice the dark areas of the direction image coincide with the values of the gradient image.

See the code for finding gradients below:

```python
def sobel(img):
    '''
    Performs sobel operation to get the gradient in x & y direction.
    Thresholds the magnitude direction which is achieved using x & y gradients
    Returns images for x-direction gradient and the thresholded magnitude direction. 
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take the derivative in x & y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Take the absolute value of the derivative or gradient
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)    
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Create a mask of 1's where the scaled gradient magnitude is within threshold
    sobel_thresh = (30,255)
    gradx = np.zeros_like(scaled_sobelx)
    gradx[(scaled_sobelx >= sobel_thresh[0]) & (scaled_sobelx <= sobel_thresh[1])] = 1
    # Get the binary image of the thresholded magnitude direction
    dir_binary = sobel_mag_dir(abs_sobelx, abs_sobely)
    return gradx, dir_binary

def sobel_mag_dir(abs_sobelx, abs_sobely, dir_thresh=(0.0, 0.3)):
    ''' Returns thesholded binary image of magitude direction from sobel gradients '''
    # Calculate absolute direction of gradient
    absgraddir = np.arctan2(abs_sobelx, abs_sobely)
    # Create binary image of thresholded gradient direction
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1
    return binary_output
```

![alt_text][image7]

#### Region of interest
***
Finally, A region of interest is applied to the result of the color selected and gradient images to eliminate noise caused by the surrounding environment (e.g. grass, adjacent vehicles, etc.)
![alt_text][image8]
***

### 4. Have lane line pixels been identified in the rectified image and fit with a polynomial?
[image9]: ./test_images/test1.jpg
[image10]: ./output_images/writeup_images/test1_histogram.png
[image11]: ./output_images/writeup_images/test1_window.png

I implemented the moving window to detect the lane line pixels and fit a polynomial to each line. The window method uses 9 rectangular windows with a window height equal to the image height divided by 9 and a defined width to search for lane pixels for each line. To illustrate this step I will use the example image below:

![alt_text][image9]

#### Histogram
***
The position of the first rectangle is found by using a histogram on the bottom half of the image to identify the x-value that contains the most white pixels for the left and right lane line. In the example histogram below, the x-values for each value would be Left: ~420  Right: ~940
![alt_text][image10]

#### Window Search and Polynomial
***
The first windows are centered at the bottom of the image on the x-values found in the historam search and searches for lane pixels. If the amount of pixels found in the rectangle is greater than `minpix`, the window is re-centered at the average x-value of the detected pixels coordinates. This continues until all windows have been searched. A polynomial is fitted to the detected lane line pixels using `np.polyfit()`. See the example result below:
![alt_text][image11]


### 5. Having identified the lane lines, has the radius of curvature of the road been estimated? And the position of the vehicle with respect to center in the lane?
[image12]: ./output_images/writeup_images/test1_rad.png

#### Radius of Curvature
The radius of curvature estimation was identified by calculating the radius of each line individualy using the radius equation and then taking the average of two radii. To report this value in meters, I first assigned a metric value per pixel in each dimension. For the y-dimension, I used 30 meters as an estimate of the maximum distance from the vehicle that the perspective transform rectifies. For the x-dimension, I divide the standard lane width of 3.7 meters by the difference between the x-intercepts of each lane line polynomial.

```python
def curvature_radius_real(y_vals, left, right):
    ''' Calculates current radius of curvature in meters '''
    # Assign metric value to pixels in each dimension
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/(right.best_fit[-1] - left.best_fit[-1]) # meters per pixel in x dimension
    
    y_eval = np.max(y_vals)
    
    # Convert best fit polynomial to real world values
    left_fit_cr = np.polyfit(y_vals*ym_per_pix, left.bestx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(y_vals*ym_per_pix, right.bestx*xm_per_pix, 2)
    
    # Calculate radius of curvature
    left.radius_of_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right.radius_of_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

#### Vehicle Position
The vehicle position is calculated by first finding the difference between the x-intercept of the lane line and the center of the image, which is assumed to be the center of the vehicle. Then, I add the differences together. This value represents the vehicle position relative to the lane center. If this value is negative, the vehicle is left of center. If positive, the vehicle is right of the lane center.

```python
def vehicle_offset(img, left, right):
    ''' Calculates difference between vehicle center and lane center '''
    vehicle_center = img.shape[1] // 2  # center of vehicle is center of image
    xm_per_pix = 3.7 / (right.bestx[-1] - left.bestx[-1])  # x-values of line nearest the vehicle
    # Calculate x-value of lane center
    left.line_base_pos = left.bestx[-1] - vehicle_center    # expects negative value
    right.line_base_pos = right.bestx[-1] - vehicle_center  #expects positive value
    vehicle_offset = left.line_base_pos + right.line_base_pos
    return vehicle_offset*xm_per_pix
```
An example of a processed image with the radius and vehicle position is shown below:
![alt_text][image12]
***

## Pipeline (video)
***
### Does the pipeline established with the test images work to process the video? Has some kind of search method been implemented to discover the position of the lines in the first images in the video stream? Has some form of tracking of the position of the lane lines been implemented?

The same pipeline that is used to process the test images is used to process the video. However, when processing a video, I use a more effiecient method of finding the lane pixels once a lane has been successfully detected. This method will search arround the previous polynomial to detect the new lane pixels which will reject outliers. This will improve the robustness of detecting the lane line from frame to frame as the position of lane lines should have little variation between frames. (See in code comments in `fit_polynomial` and `search_around_poly` for more detailed explanation). See the result of the processed video in the output file `project_video_output.mp4`.

A class `Line()` is used to keep track of each lane line and allows for a weighted average calculation of previous lines. Code shown below:

```python
##################### CREATE 'Line()' CLASS ######################
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = deque() 
        #average x values of the fitted line over the last n iterations
        self.bestx = None    
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([])  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None  
```
***
#### Sanity Checks
I used some sanity checks to find the best fit for the lane line. After the lane pixels have been detected, a polynomial has been found, and the fitted x-values have been found I implement the following sanity checks:
* check for the currently (most recent polynomial) poly-fitted coefficients. This ensures a line is currently fitted to the line. If none present, set the detected polynomial to the `current_fit'. 
* check the x-intercepts (x-values at bottom of image) of the new calculated fit  against the current fit.
* check the x-value at the top of the polynomial of the best x-values of the currently fitted polynomial against the  x-fitted values of the new detected line.

These checks will ensure that line fitted by the detected line pixels is similar to the previous lane line. I provide a +/-50 pixel window for the x-intercept and +/-60 pixels for the top x-value. I give an extra +/-10 pixels for the top because this is area where the camera will first see any curves, which can make cause the end of the lane line shift more quickly between frames compared to the x-intercept at the bottom of the image.  

The following snippet of code are the sanity inside `fit_polinomial`
```python
# Sanity checks for x-intercepts and x values 
if ((len(left.current_fit) > 1) & (len(right.current_fit) > 1)):
    # Find difference between the best 
    left.diffs = np.abs(left_fit - left.current_fit)
    right.diffs = np.abs(right_fit - right.current_fit)
    left_x_diff = np.abs(left_fitx[0] - left.bestx[0])
    right_x_diff = np.abs(right_fitx[0] - right.bestx[0])

    # Sanity checks
    # Left line - check if similar to current line
    if ((left.diffs[2] < 50) & (left_x_diff < 60)):
        left.detected = True
        left.current_fit = left_fit
        left.bestx = weighted_average(left, left_fitx)
    # New line is not similar to current, try window search for next frame 
    else:
        left.detected = False
        left.current_fit = []
        left.bestx = weighted_average(left, left_fitx)
        left.recent_xfitted.clear

    # Right Line - check if similar to current line
    if ((right.diffs[2] < 50) & (right_x_diff < 60)):
        right.detected = True
        right.current_fit = right_fit
        right.bestx = weighted_average(right, right_fitx)
    # New line is not similar to current, try window search for next frame 
    else:
        right.detected = False
        right.current_fit = []
        right.bestx = weighted_average(right, right_fitx)
        right.recent_xfitted.clear

# If no current_fit, set it as the new fit (e.g. first frame, error)         
else:
    left.detected = True
    right.detected = True
    left.current_fit = left_fit
    right.current_fit = right_fit
    # Set bestx as new x-fitted values if empty, otherwise clear queue 
    if left.bestx == None:
        left.bestx = left_fitx
    else:
        left.recent_xfitted.clear

    if right.bestx == None:
        right.bestx = right_fitx
    else:
        right.recent_xfitted.clear
```
#### Determine Best Fit
After multiple lane lines have been detected, I determine the best fit for the line by applying a weighted average to the previous 7 successfully detected lane lines. If a lane line is not detected, the weighted average will decrease the queue of x-fitted values starting with the oldest set. This would continue for multiple frames with missed lane lines until it is empty. If the queue empties, the code will revert back to the window serach function for detecting lane lines and build the queue back up. The `weighted_average()` function is shown below:

```python
def weighted_average(h, fitx, n=7):
    """ 
    Applies a weighted average to last 'n' x-fitted values.
    
    h: handle of class object
    """
    weights=[1,1,2,3,5,8,13]#,21,34,55]
    run_len = len(h.recent_xfitted)  # get current length of queue

    # Maintaining running window |f0,f1,f2,...,fn| of previous n x-values
    if fitx is not None:
        if run_len == n:  # queue is full
            h.recent_xfitted.popleft()  # [f0<--,|f1,f2,...,fn|,fn+1]
        else:             # queue not full
            run_len += 1  # add 1 to length
        h.recent_xfitted.append(fitx)  # [f0,|f1,f2,...,fn<--|,fn+1]
    else:  # reduce queue if no x-fitted values are found
        if len(h.recent_xfitted) > 0:
            h.recent_xfitted.popleft()
            run_len -= 1

    # Compute weighted average
    if len(h.recent_xfitted) > 0:
        h.bestx = np.average(h.recent_xfitted, axis=0, weights=weights[:run_len])
    return h.bestx
```
***
[video1]: ./project_video_output.mp4 "Processed Video"

Here is a link to my [Processed Video][video1]
***

## Discussion
In this section, I will explain my pipeline and any improvements I would make if I pursued this project further.

### Pipeline Explanation
***
NOTE: Before the pipeline is used, a Line() instance must be created for each lane line.
1. Create copy of input image
2. Undistort image using the calibration data obtained from `camera_calibration.ipynb` code
3. Get a "birds-eye" view of the lane
   * I do this before I perform color selection and gradient functions because the transformed will have nearly vertical lines which is an advantage when filtering the direction gradient. Also, there will be less noise caused by the scene surrounding the region being transformed, effectively making the detection of colors and gradients more robust.
4. Use HLS color spacce and Red layer of the image to find lane line pixels.
   * HLS color space improves robustness of Red layer thresholding to ensure the pixel values represent yellow or white values.
   * HLS also improves the ability to detect lane lines with varying pavement color and when shadows are present.
5. Gradients are used to find where difference in intensity values from one pixel to the next is significant.
   * This allows me to find the edges of the lane line because the contrast between the lane line and the pavement color is generally significant.
   * I used the gradient magnitude direction as a mask to filter pixels that might not be be part of the line. The direction will find pixels that are not ~mostly~ vertical. This will result in dark (filled with '0') areas where the lane lines should be which will be and-ed with the combined image of color_select and the gradient of x-dimension.
6. Find lane pixels
   * If no previous lines detected, use window search
   * If previous lines detected, search around previous polynomial
   * Get polynomial of lane line pixels and the x values of polynomial
   * If there is a previously detected line, use sanity checks to determine the best fit using a weighted average of previous 6 frames and the new x-fitted values. If the line does not pass the sanity checks, include it in the weighted average but use the window search for the next frame and clear the queue of previous x-fitted values. This will refresh the weighted average to focus on new detections which may have been caused by a sharp curve or a change in road pavement.
   * If no lines were detected at all, the new fit is set as the previous best fit and the queue is cleared.
7. Radius of curvature in meters
   * Set a metric value per pixel in each dimension using standard road width and the difference of the x-incercept of the detected lane line polynomial.
   * Fit metric-scaled polynomial
   * Use radius formula to find the radius in meters.
   * Average the two radius values to get a decent estimate of overal curve.
8. Find vehicle position relative to the lane center in meters
   * Find difference from x-intercept of each line to the center
   * Add diferences together and scale result using x-dimension scale factor
   * If result is negative, vehicle is left of lane center. If result is positive, vehicle is right of center.
9. Unwarp image using 'Minv' to display detected lane on the original image
10. Write radius of curvature and vehicle position on unwarped image

### Potential Improvements
***
* Implement more dynamic transform region for "birds-eye" view for more accurate radius of curvature calculation. This will also make the processing time slower.
* Implement a solution to help display more accurate lane region when vehicle goes over a bump in the road without causing a ripple effect on next few detections.
* Use the vehicle center to estimate where to look for the next lane line x-intercept.
* Implement a sanity checks to check variation between left and right calculations
