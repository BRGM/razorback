import razorback as rb
import datetime
import h5py
from loguru import logger
import sys

""" function to handle 'mth5' files
"""

__all__ = ["load_mth5", "clean_whitespace"]


class SuppressLogs:
    def __init__(self, level="ERROR"):
        self.level = level
        self._previous_level = None
        self.handler_id = None

    def __enter__(self):
        self._previous_level = logger._core.min_level
        logger.remove()
        self.handler_id = logger.add(sys.stderr, level=self.level)
        return self

    def __exit__(self, *args):
        logger.remove(self.handler_id)
        logger.add(sys.stderr, level=self._previous_level)


def clean_whitespace(separator):
    def clean_field(string):
        return string.replace(" ", separator)

    return clean_field


def load_mth5(filename, tag_template, clean_field=None):
    """
    return a SignalSet iterable from mth5 file

    Parameters
    ----------
    filename: str | Path
        path to the mth5 file
    tag_template: str
        template string for generating tags
        the fields used for the template are:
            - survey
            - station
            - channel
    clean_field: function
        a function that takes a string and returns a string
        use to clean the fields before passing to tag_template

    Yields
    ------
    razorback.SignalSet
        each SignalSet contains one channel of one station of one survey.
    """
    from mth5.mth5 import MTH5
    # TODO add calibrations
    if clean_field is None:
        clean_field = lambda e: e
    with MTH5().open_mth5(filename, mode="r") as m:
        survey = m.survey_group
        for station_id in survey.stations_group.groups_list:
            station = survey.stations_group.get_station(station_id)
            for run_id in station.run_summary.id:
                run = station.get_run(run_id)
                for channel_id in run.groups_list:
                    with SuppressLogs():
                        channel = run.get_channel(channel_id)
                    formated_name = tag_template.format(
                        survey=clean_field(channel.survey_id),
                        station=clean_field(
                            channel.station_metadata.model_dump(warnings=False)["id"]
                        ),
                        channel=clean_field(channel.metadata.component),
                    )
                    tags = {formated_name: 0}
                    signals = rb.SyncSignal(
                        [
                            h5py.File(channel.hdf5_dataset.file.filename)[
                                channel.hdf5_dataset.name
                            ]
                        ],
                        channel.sample_rate,
                        datetime.datetime.fromisoformat(str(channel.start)).timestamp(),
                    )
                    yield rb.SignalSet(tags, signals)
