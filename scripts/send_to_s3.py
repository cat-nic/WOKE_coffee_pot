import os
import subprocess

if 'WOKE' in os.getcwd():
    pass
else:
    os.chdir('..')

file = r'C:\Users\Paynen3\PycharmProjects\WOKE_coffee_pot\data\test_coffee_pot.jpg'
target = r's3://woke-coffee-pot/coffee_pot1.jpg'
subprocess.run(['aws',
                's3',
                'cp' ,
                f'{file}',
                f'{target}'
                ],
               # check=True,
               shell=True,
               # stdout=subprocess.PIPE
               )
