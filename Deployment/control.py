import RPi.GPIO as GPIO
import time
import cv2
import imagezmq
import socket

# GPIO setup for motor control
GPIO.setmode(GPIO.BOARD)

# Define motor pins
ena_pin = 11
in1_pin = 13
in2_pin = 15

enb_pin = 16
in3_pin = 18
in4_pin = 22

# Setup motor pins
motor_pins = [ena_pin, in1_pin, in2_pin, enb_pin, in3_pin, in4_pin]
for pin in motor_pins:
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

# Control motors based on lane angle
def control_motors(angle):
    MAX_SPEED = 100
    BASE_SPEED = 50
    ANGLE_THRESHOLD = 5.0

    speeds = [MAX_SPEED] * len(motor_pins)
 
    L_STEER_CONSTANT = 10
    R_STEER_CONSTANT = 10
    Kl = 0
    Kr = 0
 
    if angle > ANGLE_THRESHOLD: #right
        Kr = R_STEER_CONSTANT
        Kl = 0
    else:
        Kr = 0
        Kl = L_STEER_CONSTANT
 
    left_pins = [IN1, IN2]
    right_pins = [IN3, IN4]
 
    for i in range(len(left_pins)):
        speeds[i] = BASE_SPEED - int(angle * Kl)
    for i in range(len(right_pins)):
        speeds[i + len(left_pins)] = BASE_SPEED - int(angle * Kr)
 
    speeds = [max(0, min(MAX_SPEED, speed)) for speed in speeds]
 
    for i, pin in enumerate(motor_pins[1:]):  # Skip the ENA and ENB pins
        GPIO.output(pin, GPIO.HIGH if speeds[i] > 0 else GPIO.LOW)

    ENA.ChangeDutyCycle(speeds[0])
    ENB.ChangeDutyCycle(speeds[len(left_pins) + len(right_pins)])

sender = imagezmq.ImageSender(connect_to='tcp://10.202.1.197:5555')
rpi_name = socket.gethostname()
video_capture = cv2.VideoCapture(0)

try:
    while True:
        ret, image = video_capture.read()
        if ret:
            out = sender.send_image(rpi_name, image)
            print(out)
            control_motors(int(out))
        else:
            print('Error in capturing frame')

except KeyboardInterrupt:
    pass

# Stop motors and cleanup GPIO
ENA.stop()
ENB.stop()
GPIO.cleanup()
