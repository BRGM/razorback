from mth5.mth5 import MTH5
import razorback as rb
import datetime
import h5py
from loguru import logger
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logger.remove() 
logger.add(sys.stderr, level="ERROR")

""" function to handle 'mth5' files
"""

__all__ = ['load_mth5']

def load_mth5(filename, tag_template, clean_field, calibrations=None):
    """ 
    return a SignalSet iterable from mth5 file
    """
    with MTH5().open_mth5(filename, mode="r") as m:
        survey = m.survey_group
        for station_id in survey.stations_group.groups_list:
            station = survey.stations_group.get_station(station_id)
            for run_id in station.run_summary.id:
                run = station.get_run(run_id)
                for channel_id in run.groups_list:
                    channel = run.get_channel(channel_id)
                    formated_name = tag_template.format(
                        survey = clean_field(channel.survey_id),
                        station = clean_field(channel.station_metadata.model_dump()["id"]),
                        channel = clean_field(channel.metadata.component),
                    )
                    tags = {formated_name: 0}
                    signals = rb.SyncSignal(
                        [h5py.File(channel.hdf5_dataset.file.filename)[channel.hdf5_dataset.name]],
                        channel.sample_rate,
                        datetime.datetime.fromisoformat(str(channel.start)).timestamp(),
                        calibrations
                    )
                    yield rb.SignalSet(tags, signals)
