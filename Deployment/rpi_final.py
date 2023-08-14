import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import RPi.GPIO as GPIO
import time
 
tflite_model_path = r"/home/pi/Downloads/Trained_model.tflite"
interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print(input_shape)
 
# GPIO setup for motor control
GPIO.setmode(GPIO.BCM)
 
# Define motor pins
ena_pin = 12 #r
in1_pin = 11
in2_pin = 15
 
enb_pin = 13 #l
in3_pin = 18
in4_pin = 22
start = time.time()
# Setup motor pins
enable_pins = [ena_pin, enb_pin]
motor_pins = [in1_pin, in2_pin, in3_pin, in4_pin]
 
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
for pin in enable_pins:
    GPIO.setup(pin, GPIO.OUT)
 
# Initialize motor driver pins
ENA = GPIO.PWM(ena_pin, 1000)
IN1 = in1_pin
IN2 = in2_pin
ENB = GPIO.PWM(enb_pin, 1000)
IN3 = in3_pin
IN4 = in4_pin
 
ENA.start(0)
ENB.start(0)
 
GPIO.output(IN1,GPIO.LOW)
GPIO.output(IN2,GPIO.HIGH)   
GPIO.output(IN3,GPIO.LOW)
GPIO.output(IN4,GPIO.HIGH)
 
error_I = 0
 
# Control motors based on lane angle
def control_motors(error_I, angle):
    MAX_SPEED = 50  #prolly decrease it
    BASE_SPEED = 37
    ANGLE_THRESHOLD = 15.0
 
    #Scaling the angle & Integral control
    #add a negative shift
 
    speeds = [MAX_SPEED] * len(enable_pins)
 
    #L_STEER_CONSTANT = 1
    #R_STEER_CONSTANT = 1
    L_STEER_CONSTANT = 1
    R_STEER_CONSTANT = 1
    Kl = 0
    Kr = 0
    K_I = 0.1
 
    if angle > ANGLE_THRESHOLD: #right
        Kl = R_STEER_CONSTANT
        Kr= -0.25
    elif angle < -ANGLE_THRESHOLD:
        Kr = -L_STEER_CONSTANT#?????
        Kl = 0.25
 
 
    speeds[0] = BASE_SPEED + int(angle * Kl) #+ K_I*error_I
    speeds[1] = BASE_SPEED + int(angle * Kr) #+ K_I*error_I
 
    speeds = [max(0, min(MAX_SPEED, speed)) for speed in speeds]
 
    #for i, pin in enumerate(motor_pins[1:]):  # Skip the ENA and ENB pins
     #   GPIO.output(pin, GPIO.HIGH if speeds[i] > 0 else GPIO.LOW)
 
    print("Left: ",speeds[0],"Right: ",speeds[1])
    ENA.ChangeDutyCycle(speeds[1])
    ENB.ChangeDutyCycle(speeds[0])
 
# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []
 
def find_avg_angle(far_center_x , near_center_x , near_center_y ,far_center_y ):
 
 
    sum_angles = 0 
 
    #centers = [ i for i in range(-5, 6,1) ]
    #centers = [  i for i in range(-20, 21,1) ]
    centers = [ i for i in range(-10, 11,1) ]
 
    for i in centers:
 
        rad_angle = np.arctan(  ( far_center_x - (near_center_x + i)) / (near_center_y - far_center_y ) )
        angle = rad_angle*(180/np.pi)
        sum_angles += angle
        print(angle)
 
    return sum_angles/len(centers)
 
cap = cv2.VideoCapture(0)
 
try:
    while True:  # Show streamed images until Ctrl-C
        now = time.time()
        print("----------------",now-start)
        ret, image=cap.read()
 
 
 
 
        cv2.imshow('orig img', image)
 
        invimg= 255 - image
 
        cv2.imshow('inverted', invimg)
 
 
        #gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #threshold_value = 127
 
        image = invimg
 
        #image_hub.send_reply(b'hii')
        print('change')
        lanes = Lanes()
 
        small_img = cv2.resize(image, (160, 80))
 
        small_img_preprocessed = small_img  # Replace with your preprocessing logic
        small_img_preprocessed = small_img.astype(np.float32)
 
        small_img_preprocessed = np.expand_dims(small_img_preprocessed, axis=0)
 
 
        # Set the input tensor
        input_index = interpreter.get_input_details()[0]['index']
 
        interpreter.set_tensor(input_index, small_img_preprocessed)
        interpreter.invoke()
        prediction_matrix = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0] * 255.0
 
        lanes.recent_fit.append(prediction_matrix)
        print(type(prediction_matrix))
 
        print(len(lanes.recent_fit))
        #cv2.imshow('recentfit', lanes.recent_fit() )
        if len(lanes.recent_fit) > 5:
           lanes.recent_fit = lanes.recent_fit[1:]
 
        lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)
 
 
        print(len(lanes.avg_fit.shape))
        #cv2.imshow('avgfit', lanes.avg_fit() )
 
 
    # Calculate the center of the drivable area near the vehicle
        print("prediction shape : ", prediction_matrix.shape )
        #print(prediction_matrix)
        cv2.imshow('prediction', prediction_matrix)
        near_center_x = len(prediction_matrix[0]) // 2
        image_height = len(prediction_matrix)
 
        near_center_y = image_height
        print('near center: ' , near_center_x ,image_height )
 
    # Calculate the center of the detected lane image farther down the road
    # Find the indices where the values are above a certain threshold
        threshold = 0.5  # You can adjust this threshold based on your specific case
 
 
        lane_indices = np.where(prediction_matrix > threshold)[0]  #improve to get actual centre
 
        print('lane indices: ' , len(lane_indices))
        far_center_y = int(np.mean(lane_indices)) if len(lane_indices) > 0 else near_center_x #avg row 
 
 
        out =  np.where( prediction_matrix[far_center_y] > threshold)[0]
 
        far_center_x = int(np.mean( out  ))
 
        print('far center : ', far_center_x, far_center_y)
    # Calculate the angle based on the difference between these centers
        image_width = len(prediction_matrix[0])
        #max_angle = 45.0  # Maximum angle in degrees
 
 
        angle = find_avg_angle(far_center_x , near_center_x , near_center_y ,far_center_y )
        #rad_angle = np.arctan(  ( far_center_x - near_center_x) / (near_center_y - far_center_y ) )
        #angle = rad_angle*(180/np.pi)
        #lane_angle = ((far_center - near_center) / image_width ) * max_angle
 
        start_point_1 = (near_center_x, near_center_y)
        end_point_1 = (near_center_x, far_center_y)
        line_color = (0, 0, 255) 
        # Draw the line on the image
        cv2.line(prediction_matrix, start_point_1, end_point_1, line_color, thickness=2)
 
        start_point = (near_center_x, near_center_y)
        end_point = (far_center_x, far_center_y)
        line_color = (0, 0, 255) 
        # Draw the line on the image
        cv2.line(prediction_matrix, start_point, end_point, line_color, thickness=2)
 
        blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
        lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
 
        cv2.line(lane_drawn, start_point_1, end_point_1, line_color, thickness=2)
        cv2.line(lane_drawn, start_point, end_point, line_color, thickness=2)
        lane_image = cv2.resize(lane_drawn, (640, 480))
 
        lane_image = cv2.resize(lane_drawn, (640, 480))
 
        image = cv2.resize(image, (640, 480))
        lane_image = lane_image.astype(np.uint8)
 
        result = cv2.addWeighted(image, 1, lane_image, 1, 0)
        cv2.imshow('Lane Detection', result)
        if ret:
            final_angle = int(float(angle))
 
            print("Angle: ",final_angle)
            error_I += angle
            control_motors(error_I=error_I, angle = angle)
        else:
            print('Error in capturing frame')
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
 
except KeyboardInterrupt:
    print('Cleaning')
    ENA.stop()
    ENB.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
