import numpy


class LayerContinuous:
    def __init__(self, xmin, xmax = None, eps = 1.0, mu = 1.0):
        self.xmin = xmin
        self.xmax = xmax
        self.eps = eps
        self.mu = mu


class LayerDiscrete:
    def __init__(self, xmin, xmax = None, eps = 1.0, mu = 1.0):
        self.xmin = xmin
        self.xmax = xmax
        self.eps = eps
        self.mu = mu


class Probe:
    '''
    Класс для хранения временного сигнала в пробнике.
    '''

    def __init__(self, position, maxTime):
        '''
        position - положение пробника (номер ячейки).
        maxTime - максимально количество временных шагов для хранения в пробнике.
        '''
        self.position = position

        # Временные сигналы для полей E и H
        self.E = numpy.zeros(maxTime)
        self.H = numpy.zeros(maxTime)

        # Номер временного шага для сохранения полей
        self._time = 0

    def addData(self, E, H):
        '''
        Добавить данные по полям E и H в пробник.
        '''
        self.E[self._time] = E[self.position]
        self.H[self._time] = H[self.position]
        self._time += 1
