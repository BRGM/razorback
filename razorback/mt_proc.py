from numbers import Real, Complex
from .typed import Struct, ListOf, PairOf, Tensor


__all__ = ['MTProc', 'Result', 'FTParam', 'ProcParam', 'Run', 'Station', 'Sensor']


Field = Struct.Field


class Sensor(Struct):
    """
    """
    chtype = Field(str)
    azimuth = Field(Real)


class Station(Struct):
    """
    """
    UTM_x = Field(Real)
    UTM_y = Field(Real)
    hardware = Field(str)
    sensors = Field(ListOf[Sensor], ListOf[Sensor])
    survey = Field(str)
    company = Field(str)
    id = Field(str)


class Run(Struct):
    """
    """
    sampling = Field(Real)
    intervals = Field(ListOf[PairOf[Real]], ListOf[PairOf[Real]], doc=
        "an interval is a pair of timestamp: (start, stop)"
    )


class ProcParam(Struct):
    """
    """
    method = Field(str)
    param = Field(dict)
    remotes = Field(ListOf[Station], ListOf[Station])


class FTParam(Struct):
    """
    """
    window_func = Field(str)
    nb_periods = Field(Real)
    overlap = Field(Real)


class Result(Struct):
    """
    """
    rotation = Field(Real)
    frequencies = Field(ListOf[Real], ListOf[Real])
    impedance = Field(ListOf[Tensor[Complex, 2,2]], ListOf[Tensor[Complex, 2,2]])
    phase = Field(ListOf[Tensor[Real, 2,2]], ListOf[Tensor[Real, 2,2]])
    resistivity = Field(ListOf[Tensor[Real, 2,2]], ListOf[Tensor[Real, 2,2]])
    phase_tensor = Field(ListOf[Tensor[Real, 2,2]], ListOf[Tensor[Real, 2,2]])
    tipper = Field(ListOf[Tensor[Complex, 2]], ListOf[Tensor[Complex, 2]])


class MTProc(Struct):
    """
    """
    station = Field(Station)
    run = Field(Run)
    proc_param = Field(ProcParam)
    ft_param = Field(FTParam)
    result = Field(Result)
