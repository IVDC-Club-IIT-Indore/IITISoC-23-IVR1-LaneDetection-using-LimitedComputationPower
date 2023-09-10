import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import RPi.GPIO as GPIO
import time
#from servo import set_angle
 
tflite_model_path = r"/home/pi/Downloads/Trained_model.tflite"
interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print(input_shape)
 
# Set the GPIO mode to BCM
GPIO.setmode(GPIO.BCM)
 
# Define the GPIO pin to which the servo signal wire is connected
servo_pin = 18
 
# Set the frequency for the PWM signal (usually 50 Hz for servos)
pwm_frequency = 50
 
# Initialize the GPIO pin as an output
GPIO.setup(servo_pin, GPIO.OUT)
 
# Create a PWM object with the servo pin and frequency
pwm = GPIO.PWM(servo_pin, pwm_frequency)
 
# Function to set the servo angle
def set_angle(angle):
    duty_cycle = (angle / 18) + 2
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.001)  # Wait for 1 second to allow the servo to move
 
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
pwm.start(0)
# Rotate the servo to a specific angle (e.g., 90 degrees)
set_angle(180)
 
set_angle(0)
set_angle(90)
 
try:
    while True:  # Show streamed images until Ctrl-C
        now = time.time()
        #print("----------------",now-start)
        ret, image=cap.read()
 
 
 
 
        #cv2.imshow('orig img', image)
 
        #invimg= image
 
        #cv2.imshow('inverted', invimg)
 
 
        #gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #threshold_value = 127
 
        #image = invimg
 
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
        #cv2.imshow('prediction', prediction_matrix)
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
        angle -= 20
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
            set_angle(90-final_angle)
        else:
            print('Error in capturing frame')
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
 
except KeyboardInterrupt:
    print('Cleaning')
    ENA.stop()
    ENB.stop()
    pwm.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
