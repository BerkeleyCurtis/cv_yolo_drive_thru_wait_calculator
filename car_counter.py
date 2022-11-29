# Copyright (c) <2022>, <Adam Curtis>
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



# cv code from https://www.geeksforgeeks.org/how-to-capture-a-image-from-webcam-in-python/
# and obj detection from https://wellsr.com/python/object-detection-from-images-with-yolo/

import cv2 as cv
from imageai.Detection import ObjectDetection
path = '/Users/adamcurtis/Documents/855_mischief/'

# setup object detection
objects = ObjectDetection()
objects.setModelTypeAsYOLOv3()
objects.setModelPath(path + 'yolo.h5')
objects.loadModel()
  
# initialize the camera
cam_port = 1
cam = cv.VideoCapture(cam_port)
  
# Get webcam image
result, image = cam.read()
   
if result:

    # show image
    cv.imshow("webcamimage", image)
    cv.imwrite("webcamimage.png", image)
    # cv.waitKey(0)
    cv.destroyWindow("webcamimage")

    #detect objects in the image
    detected_obj = objects.detectObjectsFromImage(
    input_image=path + 'webcamimage.png',
    output_image_path= path + 'webcamimage_output.jpg'
    )

    #print out what objects and their locations
    for obj in detected_obj:
        print(obj["name"] + "-"
            +str(obj["percentage_probability"]),
            obj["box_points"])
  
else:
    print("Check yer webcam, scrub")

