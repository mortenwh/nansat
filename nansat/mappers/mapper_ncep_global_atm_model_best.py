import pytz
import netCDF4
import datetime

import numpy as np

from dateutil.parser import parse

from nansat.exceptions import WrongMapperError
from nansat.vrt import VRT


class Mapper(VRT):

    def __init__(self, filename, gdal_dataset, metadata, netcdf_dim=None, *args, **kwargs):
        """Read NCEP_Global_Atmospheric_Model_best.ncd from
        https://pae-paha.pacioos.hawaii.edu/thredds/dodsC/ncep_global/,
        and get Nansat object with arrays as close as possible to the
        given time.

        Time must be type datetime with timezone, e.g.,
        datetime.datetime(2024, 9, 10, 15, 0, 0, tzinfo=pytz.utc).
        """

        fn = ("https://pae-paha.pacioos.hawaii.edu/thredds/dodsC/"
                    "ncep_global/NCEP_Global_Atmospheric_Model_best.ncd")
        if filename != fn:
            raise WrongMapperError

        if netcdf_dim is None:
            raise WrongMapperError
        
        ds = netCDF4.Dataset(filename)
        metadata = {}
        for key in ds.ncattrs():
            metadata[key] = str(ds.getncattr(key))

        lon = ds["longitude"][:]
        lat = ds["latitude"][:]
        times = ds["time"][:]

        longitude, latitude = np.meshgrid(lon, lat)

        super(Mapper, self)._init_from_lonlat(longitude, latitude)
        self.dataset.SetMetadata(metadata)

        t0 = parse(ds["time"].units.strip("hours since "))
        t1 = ds["time"][:]
        dt = [datetime.timedelta(hours=t) for t in t1.data[:]]
        times = [t0 + tt for tt in dt]
        diff = np.array(times) - \
               netcdf_dim["time"].astype(datetime.datetime).replace(tzinfo=pytz.utc)
        time_index = np.abs(diff).argmin()

        for var, val in ds.variables.items():
            if "standard_name" in ds[var].ncattrs():
                if val.standard_name == "longitude":
                    lon = var
                if val.standard_name == "latitude":
                    lat = var

        for attr in ds.ncattrs():
            content = ds.getncattr(attr)
            if type(content) is str:
                metadata[attr] = content

        self.band_vrts = {}
        metaDict = []

        for key, val in ds.variables.items():
            data = ds[key]
            if data.shape != (t1.shape[0], longitude.shape[0], longitude.shape[1]):
                continue
            self.band_vrts[key] = VRT.from_array(
                data[time_index, :, :].filled(fill_value=np.nan))
            key_metadict = {}
            for attr in ds[key].ncattrs():
                key_metadict[attr] = ds[key].getncattr(attr)
            key_metadict["name"] = key
            key_metadict["time"] = times[time_index].isoformat()
            metaDict.append({
                'src': {
                    'SourceFilename': self.band_vrts[key].filename,
                    'SourceBand': 1
                },
                'dst': key_metadict,
            })

        self.create_bands(metaDict)

        #self.fix_global_metadata
