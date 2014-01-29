# Name:         mapper_metno_hires_seaice.py
# Purpose:      Nansat mapping for high resolution sea ice from met.no Thredds server
# Authors:      Knut-Frode Dagestad
# Licence:      This file is part of NANSAT. You can redistribute it or modify
#               under the terms of GNU General Public License, v.3
#               http://www.gnu.org/licenses/gpl-3.0.html

# High resolution (1 km) manual ice concentration, based on SAR imagery
# http://thredds.met.no/thredds/catalog/myocean/siw-tac/siw-metno-svalbard/
#
# Mapper may be called with full URL:
#   'http://thredds.met.no/thredds/dodsC/myocean/siw-tac/siw-metno-svalbard/2014/01/ice_conc_svalbard_201401091500.nc
# or with keyword (fake filename):
#   'metno_hires_seaice:20140109'
#
# The latter syntax will construct the URL, and will return the closest available data within +/- 3 days

import sys
import os
import urllib2
from datetime import datetime, timedelta
from vrt import *
import osr

try:
    from osgeo import gdal
except ImportError:
    import gdal

import mapper_generic as mg


class Mapper(mg.Mapper):
    ''' Create VRT with mapping of WKV for Met.no seaice '''

    def __init__(self, fileName, gdalDataset, gdalMetadata, **kwargs):
        ''' Create VRT '''

        try:
            iceFolderName = kwargs['iceFolder']
        except:
            #iceFolderName = '/vol/istjenesten/data/metnoCharts/'
            iceFolderName = '/vol/data/metnoCharts/'
            
        keywordBase = 'metno_local_hires_seaice'
        
        if fileName[0:len(keywordBase)] != keywordBase:
            raise AttributeError("Wrong mapper")

        keywordTime = fileName[len(keywordBase)+1:]
        requestedTime = datetime.datetime.strptime(keywordTime, '%Y%m%d')
        # Search for nearest available file, within the closest 3 days
        foundDataset = False
        for deltaDay in [0, -1, 1, -2, 2, -3, 3]:
            validTime = requestedTime + timedelta(days=deltaDay) + timedelta(hours=15)
            fileName = iceFolderName + 'ice_conc_svalbard_' + validTime.strftime('%Y%m%d1500.nc')
            if os.path.exists(fileName):
                print 'Found file:'
                print fileName
                gdalDataset = gdal.Open(fileName)
                gdalMetadata = gdalDataset.GetMetadata()
                mg.Mapper.__init__(self, fileName, gdalDataset, gdalMetadata)
                foundDataset = True
                # Modify GeoTransform from netCDF file - otherwise a shift is seen! 
                self.dataset.SetGeoTransform(
                    (-1243508 -1000, 1000, 0, -210526 -7000, 0, -1000))
                break # Data is found for this day

        if foundDataset is False:
            AttributeError("No local Svalbard-ice files available")
            sys.exit()
