import RPi.GPIO as GPIO
import time
import cv2
import imagezmq
import socket
 
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
    MAX_SPEED = 100  #prolly decrease it
    BASE_SPEED = 50
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
        Kr= -1.85
    elif angle < -ANGLE_THRESHOLD:
        Kr = -L_STEER_CONSTANT#?????
        Kl = 1.85 
 
 
 
    speeds[0] = BASE_SPEED + int(angle * Kl) #+ K_I*error_I
    speeds[1] = BASE_SPEED + int(angle * Kr) #+ K_I*error_I
 
    speeds = [max(0, min(MAX_SPEED, speed)) for speed in speeds]
 
    #for i, pin in enumerate(motor_pins[1:]):  # Skip the ENA and ENB pins
     #   GPIO.output(pin, GPIO.HIGH if speeds[i] > 0 else GPIO.LOW)
 
    print("Left: ",speeds[0],"Right: ",speeds[1])
    ENA.ChangeDutyCycle(speeds[1])
    ENB.ChangeDutyCycle(speeds[0])
 
sender = imagezmq.ImageSender(connect_to='tcp://10.202.5.217:5555')
rpi_name = socket.gethostname()
video_capture = cv2.VideoCapture(0)
 
try:
    while True:
        now = time.time()
        print("----------------",now-start)
        ret, image = video_capture.read()
        if ret:
            out = sender.send_image(rpi_name, image)
            #print(out)
            angle = int(float(out.decode('utf-8')))
 
            print("Angle: ",angle)
            #out = '0'
            error_I += angle
            control_motors(error_I=error_I, angle = angle)
        else:
            print('Error in capturing frame')
 
except KeyboardInterrupt:
    print('Cleaning')
    ENA.stop()
    ENB.stop()
    GPIO.cleanup()
 
