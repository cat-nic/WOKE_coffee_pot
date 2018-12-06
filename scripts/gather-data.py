try:  # Just to get rid of warnings on Windows machines
    from picamera import PiCamera
except ImportError:
    PiCamera = None
import os
import subprocess
import logging
import datetime
import argparse
if 'WOKE' in os.getcwd():
    pass
else:
    os.chdir('..')

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--state", required=True,
	help="State of the coffee pot - whether full, half, or empty")
ap.add_argument("-u", "--upload", required=True,
	help="Boolean whether or not to upload to s3 or only store locally")
args = vars(ap.parse_args())

state = args['state']
upload = args['upload']

logging.basicConfig(filename=os.getcwd() + "\\s3_send_log.log")

# AWS CLI NEEDS TO BE INSTALLED WHICH CAN BE DONE WITH PIP
# pip install awscli --upgrade --user
def simple_send(img, target):
    # """:param img: ex. r'C:\Users\Paynen3\PycharmProjects\WOKE_coffee_pot\data\test_coffee_pot.jpg'
    # :param target: ex.  r's3://woke-coffee-pot/coffee_pot1.jpg'
    # :return:"""
    try:
        subprocess.run(['aws',
                    's3',
                    'cp',
                    f'{img}',
                    f'{target}'
                    ],
                   # check=True,
                   shell=True,
                   # stdout=subprocess.PIPE
                   )
        return 1
    except Exception as e:
        logging.exception(e)
        return 0

def take_pic_and_send():
    """
    Takes a quick picture and uploads to relevant s3 bucket folder
    :return:
    """
    try:
        camera = PiCamera()
        ts = datetime.datetime.now().strftime(format="%Y-%m-%d-%H-%M-%S")
        img = "data/pic-{}.jpg".format(ts)
        camera.capture(img)
        if upload:
            try:
                target = r's3://woke-coffee-pot/{}/{}'.format(state, img)
                simple_send(img, target)
                os.remove(img)
            except Exception as e:
                logging.exception(e)
                return 0
        return 1
    except Exception as e:
        logging.exception(e)
        return 0

# TODO: MAYBE DESIGN THIS TO BE IN A LOOP
