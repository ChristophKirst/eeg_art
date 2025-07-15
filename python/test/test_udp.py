#%%

import time
import numpy as np

from pythonosc.udp_client import SimpleUDPClient
ip = '127.0.0.1'
port = 8200
client = SimpleUDPClient(ip, port)

for i in range(1000000):
    print(i)
    p = np.random.random()
    client.send_message('/pitch', p)
    time.sleep(0.02)

#%%


#%%


