# Module to parse GPS Dongle input into latitude and longitude coordinates

from gps import *
import time


#reads in gps data and extracts lat/long info
def getLocation(gpsd):   
    data = gpsd.next()
    if data['class'] == 'TPV':
        lon = getattr(data, 'lon', "Unknown")
        lat = getattr(data, 'lat', "Unknown")
        return lon, lat


active = True
gpsd = gps(mode=WATCH_ENABLE|WATCH_NEWSTYLE)


#testing gps parse code
try:
    while active:       
        coords = getLocation(gpsd)
        
        #Handles faulty readings
        if (coords == None):
            pass
        else:
            longitude = coords[0]
            latitude = coords[1]
            print("Longitude: ", longitude)
            print("Latitude: ", latitude)

        #adjust sleep time for frequency of readings
        time.sleep(2)
except (KeyboardInterrupt):
    active = False
    print("Exiting")